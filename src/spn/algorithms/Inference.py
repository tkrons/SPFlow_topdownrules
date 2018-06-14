'''
Created on March 21, 2018

@author: Alejandro Molina
'''
import numpy as np
from scipy.special import logsumexp

from spn.structure.Base import Product, Sum, Leaf

EPSILON = 0.000000000000001
LOG_ZERO = -300


def compute_likelihood_children(node, data, dtype, node_likelihood, lls_matrix):
    llchildren = np.zeros((data.shape[0], len(node.children)), dtype=dtype)

    # TODO: parallelize here
    for i, c in enumerate(node.children):
        llchild = likelihood(c, data, dtype=dtype, node_likelihood=node_likelihood, lls_matrix=lls_matrix)
        assert llchild.shape[0] == data.shape[0]
        assert llchild.shape[1] == 1
        llchildren[:, i] = llchild[:, 0]

    return llchildren


def prod_log_likelihood(node, data, dtype, node_likelihood, lls_matrix):
    llchildren = compute_likelihood_children(node, data, dtype, node_likelihood, lls_matrix)
    return np.sum(llchildren, axis=1).reshape(-1, 1)


def prod_likelihood(node, data, dtype, node_likelihood, lls_matrix):
    llchildren = compute_likelihood_children(node, data, dtype, node_likelihood, lls_matrix)
    return np.prod(llchildren, axis=1).reshape(-1, 1)


def sum_log_likelihood(node, data, dtype, node_likelihood, lls_matrix):
    llchildren = compute_likelihood_children(node, data, dtype, node_likelihood, lls_matrix)

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    return logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)


def sum_likelihood(node, data, dtype, node_likelihood, lls_matrix):
    llchildren = compute_likelihood_children(node, data, dtype, node_likelihood, lls_matrix)

    assert np.isclose(np.sum(node.weights), 1.0), "unnormalized weights {} for node {}".format(node.weights, node)

    b = np.array(node.weights, dtype=dtype)

    return np.dot(llchildren, b).reshape(-1, 1)


_node_log_likelihood = {Sum: sum_log_likelihood, Product: prod_log_likelihood}
_node_likelihood = {Sum: sum_likelihood, Product: prod_likelihood}


def add_node_likelihood(node_type, lambda_func):
    log_lambda = lambda node, data, **args: np.log(lambda_func(node, data, **args))

    _node_likelihood[node_type] = lambda_func
    _node_log_likelihood[node_type] = log_lambda


_node_mpe_likelihood = {}


def add_node_mpe_likelihood(node_type, lambda_func):
    _node_mpe_likelihood[node_type] = lambda_func


def likelihood(node, data, dtype=np.float64, node_likelihood=_node_likelihood, lls_matrix=None):
    assert len(data.shape) == 2, "data must be 2D, found: {}".format(data.shape)

    t_node = type(node)
    if t_node in node_likelihood:
        ll = node_likelihood[t_node](node, data, dtype=dtype, node_likelihood=node_likelihood, lls_matrix=lls_matrix)
        assert ll.shape[1] == 1
        assert ll.shape[0] == data.shape[0]
        if lls_matrix is not None:
            lls_matrix[:, node.id] = ll[:, 0]
        return ll
    else:
        raise Exception('Node type unknown: ' + str(type(node)))


def log_likelihood(node, data, dtype=np.float64, node_log_likelihood=_node_log_likelihood, lls_matrix=None):
    return likelihood(node, data, dtype=dtype, node_likelihood=node_log_likelihood, lls_matrix=lls_matrix)


# TODO: test this function super thorougly


def mpe_likelihood(node, data, log_space=True, dtype=np.float64, context=None, node_mpe_likelihood=_node_mpe_likelihood,
                   lls_matrix=None):
    #
    # for leaves it should be the same, marginalization is being taken into account
    if node_mpe_likelihood is not None:
        t_node = type(node)
        if t_node in node_mpe_likelihood:
            ll = node_mpe_likelihood[t_node](node, data, log_space=log_space, dtype=dtype, context=context,
                                             node_mpe_likelihood=node_mpe_likelihood)
            if lls_matrix is not None:
                assert ll.shape[1] == 1, ll.shape[1]
                lls_matrix[:, node.id] = ll[:, 0]
            return ll

    is_product = isinstance(node, Product)

    is_sum = isinstance(node, Sum)

    # print('nnode id', node.id, is_product, is_sum)

    if not (is_product or is_sum):
        raise Exception('Node type unknown: ' + str(type(node)))

    llchildren = np.zeros((data.shape[0], len(node.children)), dtype=dtype)

    # TODO: parallelize here
    for i, c in enumerate(node.children):
        llchild = mpe_likelihood(c, data, log_space=True, dtype=dtype, context=context,
                                 node_mpe_likelihood=node_mpe_likelihood,
                                 lls_matrix=lls_matrix)
        assert llchild.shape[0] == data.shape[0]
        assert llchild.shape[1] == 1
        llchildren[:, i] = llchild[:, 0]

    if is_product:
        ll = np.sum(llchildren, axis=1).reshape(-1, 1)

        if not log_space:
            ll = np.exp(ll)

    elif is_sum:
        #
        # this actually computes the weighted max
        # b = np.array(node.weights, dtype=dtype)

        # ll = logsumexp(llchildren, b=b, axis=1).reshape(-1, 1)
        w_lls = llchildren + np.log(node.weights)
        # print(node.id, 'WLLs', w_lls, llchildren, np.log(node.weights))
        ll = np.max(w_lls, axis=1, keepdims=True)

        if not log_space:
            ll = np.exp(ll)
    else:
        raise Exception('Node type unknown: ' + str(type(node)))

    assert ll.shape[1] == 1

    if lls_matrix is not None:
        lls_matrix[:, node.id] = ll[:, 0]

    return ll


def conditional_log_likelihood(node_joint, node_marginal, data, log_space=True, dtype=np.float64):
    result = log_likelihood(node_joint, data, dtype) - log_likelihood(node_marginal, data, dtype)
    if log_space:
        return result

    return np.exp(result)
