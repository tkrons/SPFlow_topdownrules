"""
Created on March 25, 2018

@author: Alejandro Molina
"""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise
from sklearn.tree import _tree, DecisionTreeClassifier, plot_tree, export_text
from spn.structure.Base import Rule, Condition
from spn.structure.leaves.parametric.Parametric import Categorical

import warnings
from spn.algorithms.splitting.Base import split_data_by_clusters, preproc
import logging

logger = logging.getLogger(__name__)
_rpy_initialized = False


def init_rpy():
    global _rpy_initialized
    if _rpy_initialized:
        return
    _rpy_initialized = True

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    import os

    path = os.path.dirname(__file__)
    with open(path + "/mixedClustering.R", "r") as rfile:
        code = "".join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()


def get_split_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans


def get_split_rows_TSNE(n_clusters=2, pre_proc=None, ohe=False, seed=17, verbose=10, n_jobs=-1):
    # https://github.com/DmitryUlyanov/Multicore-TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import os

    ncpus = n_jobs
    if n_jobs < 1:
        ncpus = max(os.cpu_count() - 1, 1)

    def split_rows_KMeans(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)
        kmeans_data = TSNE(n_components=3, verbose=verbose, n_jobs=ncpus, random_state=seed).fit_transform(data)
        clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(kmeans_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_KMeans

def get_split_rows_KM_RuleClustering(model, k=2, rand_state=None, debug=None, pre_proc=None, ohe=False,):
    #todo return listof(tuple(data, indices, weight))
    # todo only works for one-hot-encoding... categorical data (n>2) does not work
    def tree_to_rule(tree, feature_names, ds_context):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined'
            for i in tree_.feature
        ]

        def recurse(node, depth, rule=[]):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                feature = feature_name[node]

                threshold = tree_.threshold[node]
                # left <= threshhold < right
                child_l, child_r = tree_.children_left[node], tree_.children_right[node]
                impure_l, impure_r = tree_.impurity[child_l], tree_.impurity[child_r]
                if ds_context.parametric_types[feature] == Categorical:
                    if impure_l < impure_r: # go left
                        # assuming binary!
                        rule.append(Condition(feature, np.equal, int(np.floor(threshold))))
                        return recurse(child_l, depth + 1, rule)
                    else: # go right
                        rule.append(Condition(feature, np.equal, int(np.ceil(threshold))))
                        return recurse(child_r, depth + 1, rule)
                else: #todo nonbinary case? then broken?
                    if impure_l < impure_r: # go left
                        rule.append(Condition(feature, np.less_equal, threshold))
                        return recurse(child_l, depth + 1, rule)
                    else: # go right
                        rule.append(Condition(feature, np.greater, threshold))
                        return recurse(child_r, depth + 1, rule)
            else:
                return rule
        return Rule(recurse(0, 1,))


    def split_rows_RuleClustering(local_data, ds_context, scope,):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        #https://stackoverflow.com/a/39772170/5595684
        km = KMeans(k, random_state=rand_state)
        km_clusters = km.fit_predict(data)
        lab, count = np.unique(km.labels_, return_counts=True)
        #inverse weight classes, todo test if this works ok
        N = len(data)
        lab_wgt = {lab: (N - count) / N for lab, count in zip(lab, count)}
        W = [lab_wgt[lab] for lab in km.labels_]

        if model == 'stump':
            dtc = DecisionTreeClassifier(random_state=rand_state, max_depth=1,).fit(data, km.labels_, sample_weight=W)
            # dtc.cost_complexity_pruning_path()
            left_rule = tree_to_rule(dtc, scope, ds_context)
        elif model == 'tree':
            dtc = DecisionTreeClassifier(
                random_state=rand_state, max_depth=None, ccp_alpha=0.05, min_impurity_split=0.01 #max_leaf_nodes=2*10**(self.k+1)
                                         ).fit(data, km.labels_, sample_weight = W
                                               )
            # dtc.cost_complexity_pruning_path()
            left_rule = tree_to_rule(dtc, scope, ds_context)
        elif model == 'm-estimate':
            raise ValueError('Not implemented')
        else:
            raise ValueError(str(model) + ' unknown model type')

        if debug:
            import matplotlib as plt #todo remove when everythings working
            dt_labels = dtc.predict(data)
            # plot_tree(dtc)
            # plt.show()
            print(export_text(dtc.tree_.value))

            if data.shape[1] == 2:
                fig, ax = plt.subplots()
                colors = np.full(dt_labels.shape, 'blue', dtype=object)
                np.putmask(colors, dt_labels.astype(bool), 'green')
                colors[km.labels_ != dt_labels] = 'black'
                #plot rule:
                assert len(left_rule) <= 2
                for cond in left_rule:
                    if cond['feature'] == 0:
                        ax.axvline(cond['threshhold'], )
                    else:
                        ax.axhline(cond['threshhold'])
                ax.scatter(data[:, 0], data[:, 1], c=colors)
                plt.show()

        # todo try out rule clusters
        # rule_clusters = rule.apply(data)
        split = split_data_by_clusters(data, km_clusters, scope, rows=True)
        assert len(split) == 2
        right_rule = left_rule.negate()
        return split, (left_rule, right_rule)
    return split_rows_RuleClustering

def get_split_rows_RuleClustering(model, k=2, rand_state=None, debug=None, pre_proc=None, ohe=False,):
    def split_rows_RuleClustering(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        # clusters = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(data)
        if model == 'stump':
            # choose dim and cutoff to maximize (some) distance?
            clusters = 123

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    raise ValueError()
    return split_rows_RuleClustering

def get_split_rows_DBScan(eps=2, min_samples=10, pre_proc=None, ohe=False):
    def split_rows_DBScan(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_DBScan


def get_split_rows_Gower(n_clusters=2, pre_proc=None, seed=17):
    from rpy2 import robjects

    init_rpy()

    def split_rows_Gower(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, False)

        try:
            df = robjects.r["as.data.frame"](data)
            clusters = robjects.r["mixedclustering"](df, ds_context.distribution_family, n_clusters, seed)
            clusters = np.asarray(clusters)
        except Exception as e:
            np.savetxt("/tmp/errordata.txt", local_data)
            logger.info(e)
            raise e

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_Gower


def get_split_rows_GMM(n_clusters=2, pre_proc=None, ohe=False, seed=17, max_iter=100, n_init=2, covariance_type="full"):
    """
    covariance_type can be one of 'spherical', 'diag', 'tied', 'full'
    """

    def split_rows_GMM(local_data, ds_context, scope):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        estimator = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=seed,
        )

        clusters = estimator.fit(data).predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_GMM
