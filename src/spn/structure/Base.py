"""
Created on March 20, 2018

@author: Alejandro Molina
"""
import numpy as np
import collections
from collections import deque, OrderedDict
import logging
from pandas.core.generic import NDFrame

logger = logging.getLogger(__name__)


class Node(object):
    def __init__(self):
        self.id = 0
        self.scope = []

    @property
    def name(self):
        return "%sNode_%s" % (self.__class__.__name__, self.id)

    @property
    def parameters(self):
        raise Exception("Not Implemented")

    def __repr__(self):
        return self.name

    def __rmul__(self, weight):
        assert type(weight) == int or type(weight) == float
        self._tmp_weight = weight
        return self

    def __mul__(self, node):
        assert isinstance(node, Node)
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert len(set(node.scope).intersection(set(self.scope))) == 0, "children's scope is not disjoint"
        result = Product()
        result.children.append(self)
        result.children.append(node)
        result.scope.extend(self.scope)
        result.scope.extend(node.scope)
        assign_ids(result)
        return result

    def __add__(self, node):
        assert isinstance(node, Node)
        assert hasattr(node, "_tmp_weight"), "right node has no weight"
        assert hasattr(self, "_tmp_weight"), "left node has no weight"
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert set(node.scope) == (set(self.scope)), "children's scope are not the same"

        from numpy import isclose

        assert isclose(
            1.0, self._tmp_weight + node._tmp_weight
        ), "unnormalized weights, maybe trying to add many nodes at the same time?"

        result = Sum()
        result.children.append(self)
        result.weights.append(self._tmp_weight)
        result.children.append(node)
        result.weights.append(node._tmp_weight)
        result.scope.extend(self.scope)
        result._tmp_weight = self._tmp_weight + node._tmp_weight
        assign_ids(result)
        return result


class Sum(Node):
    def __init__(self, weights=None, children=None, rule=None):
        Node.__init__(self)
        if weights is None:
            weights = []
        self.weights = weights
        self.rule = rule
        self.children_rules = {}

        if children is None:
            children = []
        self.children = children

    @property
    def parameters(self):
        sorted_children = sorted(self.children, key=lambda c: c.id)
        params = [(n.id, self.weights[i]) for i, n in enumerate(sorted_children)]
        return tuple(params)


class Product(Node):
    def __init__(self, children=None, rule=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children
        self.rule = rule
        self.children_rules = {}

    @property
    def parameters(self):
        return tuple(map(lambda n: n.id, sorted(self.children, key=lambda c: c.id)))


class Leaf(Node):
    def __init__(self, scope=None, rule=None):
        Node.__init__(self)
        self.rule = rule
        if scope is not None:
            if type(scope) == int:
                self.scope.append(scope)
            elif type(scope) == list:
                self.scope.extend(scope)
            else:
                raise Exception("invalid scope type %s " % (type(scope)))


class Context:
    def __init__(self, meta_types=None, domains=None, parametric_types=None, feature_names=None):
        self.meta_types = meta_types
        self.domains = domains
        self.parametric_types = parametric_types
        self.feature_names = feature_names

        if meta_types is None and parametric_types is not None:
            self.meta_types = []
            for p in parametric_types:
                self.meta_types.append(p.type.meta_type)

    def get_meta_types_by_scope(self, scopes):
        return [self.meta_types[s] for s in scopes]

    def get_domains_by_scope(self, scopes):
        return [self.domains[s] for s in scopes]

    def get_parametric_types_by_scope(self, scopes):
        return [self.parametric_types[s] for s in scopes]

    def add_domains(self, data):
        assert len(data.shape) == 2, "data is not 2D?"
        assert data.shape[1] == len(self.meta_types), "Data columns and metatype size doesn't match"

        from spn.structure.StatisticalTypes import MetaType

        domain = []

        for col in range(data.shape[1]):
            feature_meta_type = self.meta_types[col]
            min_val = np.nanmin(data[:, col])
            max_val = np.nanmax(data[:, col])
            domain_values = [min_val, max_val]

            if feature_meta_type == MetaType.REAL or feature_meta_type == MetaType.BINARY:
                domain.append(domain_values)
            elif feature_meta_type == MetaType.DISCRETE:
                domain.append(np.arange(domain_values[0], domain_values[1] + 1, 1))
            else:
                raise Exception("Unkown MetaType " + str(feature_meta_type))

        self.domains = np.asanyarray(domain)

        return self

class Condition(tuple):
    '''
    immutable tuple
    :param var variable name
    :param op np function of comparison
    :param threshold value used as second argument of op: IF op( x_var, threshhold ) THEN true
    '''
    from operator import itemgetter
    __slots__ = []

    def __new__(cls, var, op, threshold):
        return tuple.__new__(cls, (var, op, threshold))

    def __getnewargs__(self): # needed for pickling the class
        return self.var, self.op, self.threshold

    var = property(itemgetter(0))
    op = property(itemgetter(1))
    threshold = property(itemgetter(2))

    def apply(self, x):
        return self.op(x, self.threshold)

    def _merge_conditions(self, other, domains = None):
        var = self.var
        op = self.op
        if self.var == other.var and self.op == other.op:
            if op == np.equal:
                if self.threshold != other.threshold:
                    raise ValueError('would result in impossible rule')
                else: # self.threshold == other.threshold
                    return var, op, self.threshold

            mergeable = [np.less_equal, np.less, np.greater_equal, np.greater]
            if self.op in mergeable:
                take_own_threshold = self.op(self.threshold, other.threshold)
                if not take_own_threshold:
                    threshold = other.threshold
                else:
                    threshold = self.threshold

            elif self.op == np.equal and self.threshold == other.threshold:
                threshold = self.threshold #threshold is the same
            else:
                # res.threshold = self.threshold
                # res2 = res.copy()
                # res2.threshold = other.threshold
                # return [res, res2]
                return False
            return var, op, threshold
        elif op == np.not_equal and other.op == np.not_equal:
            if len(domains[var]) == 2:
                if self.threshold != other.threshold:
                    raise ValueError('would be impossible')
            elif self.threshold == other.threshold:
                return var, op, self.threshold
            else:
                return False #keep both
        elif (op == np.not_equal and other.op == np.equal) or (op == np.equal and other.op == np.not_equal):
            if len(domains[var]) == 2:
                if self.threshold == other.threshold:
                    raise ValueError('would result in impossible rule')
                else:  # rules are identical x==0 equals x!=1
                    threshold = (op == np.not_equal) ^ self.threshold #XOR
                    return var, np.equal, threshold

        else:
            return False

    def get_similar_conditions(self, var):
        if self.var == var:
            return [self]

    def __eq__(self, other):
        if isinstance(other, Condition):
            if self.var == other.var and self.op == other.op and self.threshold == other.threshold:
                return True
        else: return False
    def __repr__(self):
        if self.op == np.equal:
            op = '='
        elif self.op == np.less_equal:
            op = '<='
        elif self.op == np.less:
            op = '<'
        elif self.op == np.greater:
            op = '>'
        elif self.op == np.greater_equal:
            op = '>='
        elif self.op == np.not_equal:
            op = '!='
        else:
            op = str(self.op)
        return str([self.var, str(op), self.threshold])
    def __str__(self):
        return self.__repr__()
    def __hash__(self):
        return hash(self.__repr__())



class Rule(tuple):
    from operator import itemgetter
    __slots__ = []

    def __new__(cls, conditions=[],):
        if isinstance(conditions, Condition):
            return tuple.__new__(cls, tuple([conditions])) # tuple constructor doesnt want to do: ((element)) instead (element)
        elif len(conditions) == 0:
            return tuple.__new__(cls, tuple(conditions))
        elif isinstance(conditions[0], Condition):
            assert len(set(conditions)) == len(conditions)
            return tuple.__new__(cls, tuple(conditions))
        else:
            raise ValueError('Invalid conditions:' + str(conditions))


    # _conditions = property(itemgetter(0))

    def get_similar_conditions(self, var):
        res = []
        for i, c in enumerate(self):
            if c.var == var:
                res.append((i, c))
        return res

    def negate(self):
        conds = []
        # opposite = {np.equal: np.not_equal, np.not_equal: np.equal,
        #             np.greater: np.less_equal, np.greater_equal: np.less,
        #             np.less: np.greater_equal, np.less_equal: np.greater}
        for c in self:
            conds.append(Condition(c.var, c.op, 1 - c.threshold))
        return Rule(conds)

    def merge(self, other, ds_context=None):
        new_R = tuple()
        if isinstance(other, Rule):
            other_remaining = dict(zip(range(len(other)), other))
            for c in self:
                # similar = other.get_similar_conditions(c.var)
                # if similar:
                    # for i, oc in similar:
                for i, oc in enumerate(other):
                    if c == oc: # dont merge
                        other_remaining.pop(i)
                    elif c.var == oc.var:
                        # try to merge conditions, otherwise append both
                        try:
                            merged = c._merge_conditions(oc, domains=ds_context.domains)
                        except ValueError:
                            merged = c
                        if merged:
                            new_R += (Condition(*merged),)
                            other_remaining.pop(i)
                            break
                else: # no similar condition
                    new_R += (c,)
            new_R += tuple(other_remaining.values())
            return Rule(new_R)
        elif isinstance(other, Condition):
            new_c = list(self)
            similar = self.get_similar_conditions(other.var)
            if similar:
                assert len(similar) == 1
                i, similar = similar[0]
                try:
                    merged = similar._merge_conditions(other)
                except ValueError:
                    merged = similar
                new_c.pop(i)
                new_c.append(merged)
            else:
                new_c.append(other)
            return Rule(new_c)
        else:
            raise ValueError(other)




    def apply(self, data, head=None, value_dict=None):
        '''assume only AND conjunctions for now'''
        bool_vecs = []
        if isinstance(data, dict) or isinstance(data, list) or isinstance(data, NDFrame):
            if head:
                all_conditions = list(self) + [head]
            else:
                all_conditions = self
            for c in all_conditions:
                if isinstance(c.threshold, str):
                    assert value_dict
                    varindex = list(data.columns).index(c.var)
                    var_attributes = value_dict[varindex][2]
                    for k,v in var_attributes.items():
                        if v == c.threshold:
                            threshold = k
                            break
                else:
                    threshold = c.threshold
                bool_vecs.append(c.op(data[c.var], threshold))
        results = np.all(bool_vecs, axis=0)
        # if head:
        #     results[results==True] = head
        #     results[results==False] = np.NaN
        return results




    def __eq__(self, other):
        if isinstance(other, Rule):
            union = set(self).union(set(other))
            if len(union) == len(self) and len(union) == len(other):
                return True
        else:
            return False

    def __hash__(self):
        return hash(frozenset(self))

    def __ne__(self, other):
        return not self.__eq__(other)
    # def __repr__(self):
    #     s = '['
    #     for c in self:
    #         s += str(c) + ', '
    #     return s + ']'
    # def __str__(self):
    #     return self.__repr__()
    # def __len__(self):
    #     return len(self._conditions)




def get_number_of_edges(node):
    return sum([len(c.children) for c in get_nodes_by_type(node, (Sum, Product))])


def get_number_of_nodes(spn, node_type=Node):
    return len(get_nodes_by_type(spn, node_type))


def get_parents(node, includ_pos=True):
    parents = OrderedDict({node: []})
    for n in get_nodes_by_type(node):
        if not isinstance(n, Leaf):
            for i, c in enumerate(n.children):
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                if includ_pos:
                    parent_list.append((n, i))
                else:
                    parent_list.append(n)
    return parents


def get_depth(node):
    node_depth = {}

    def count_layers(node):
        ndepth = node_depth.setdefault(node, 1)

        if hasattr(node, "children"):
            for c in node.children:
                node_depth.setdefault(c, ndepth + 1)

    bfs(node, count_layers)

    return max(node_depth.values())


def rebuild_scopes_bottom_up(node):
    # this function is not safe (updates in place)

    for n in get_topological_order(node):
        if isinstance(n, Leaf):
            continue

        new_scope = set()
        for c in n.children:
            new_scope.update(c.scope)
        n.scope = list(new_scope)

    return node


def bfs(root, func):
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)


def get_topological_order(node):
    nodes = get_nodes_by_type(node)

    parents = OrderedDict({node: []})
    in_degree = OrderedDict()
    for n in nodes:
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    S = deque()  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            S.appendleft(u)

    L = []  # Empty list that will contain the sorted elements

    while S:
        n = S.pop()  # remove a node n from S
        L.append(n)  # add n to tail of L

        for m in parents[n]:  # for each node m with an edge e from n to m do
            in_degree_m = in_degree[m] - 1  # remove edge e from the graph
            in_degree[m] = in_degree_m
            if in_degree_m == 0:  # if m has no other incoming edges then
                S.appendleft(m)  # insert m into S

    assert len(L) == len(nodes), "Graph is not DAG, it has at least one cycle"
    return L


def get_topological_order_layers(node):
    nodes = get_nodes_by_type(node)

    parents = OrderedDict({node: []})
    in_degree = OrderedDict()
    for n in nodes:
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    layer = []  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            layer.append(u)

    L = [layer]  # add first layer

    added_nodes = len(layer)
    while True:
        layer = []

        for n in L[-1]:
            for m in parents[n]:  # for each node m with an edge e from n to m do
                in_degree_m = in_degree[m] - 1  # remove edge e from the graph
                in_degree[m] = in_degree_m
                if in_degree_m == 0:  # if m has no other incoming edges then
                    layer.append(m)  # insert m into layer

        if len(layer) == 0:
            break

        added_nodes += len(layer)
        L.append(layer)

    assert added_nodes == len(nodes), "Graph is not DAG, it has at least one cycle"
    return L


def get_nodes_by_type(node, ntype=Node):
    assert node is not None

    result = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)

    return result


def get_node_types(node, ntype=Node):
    assert node is not None

    result = set()

    def add_node(node):
        if isinstance(node, ntype):
            result.add(type(node))

    bfs(node, add_node)

    return result


def assign_ids(node, ids=None):
    if ids is None:
        ids = {}

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)

        node.id = ids[node]

    bfs(node, assign_id)
    return node


def eval_spn_bottom_up(node, eval_functions, all_results=None, debug=False, **args):
    """
    Evaluates the spn bottom up


    :param node: spn root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, args**) for leave nodes and (node, [children results], args**)
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node. It is used to store intermediate results so that non-tree graphs can be computed in O(n) size of the network
    :param debug: whether to present progress information on the evaluation
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
    """

    nodes = get_topological_order(node)

    if debug:
        from tqdm import tqdm

        nodes = tqdm(list(nodes))

    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
        node_type._is_leaf = issubclass(node_type, Leaf)
    leaf_func = eval_functions.get(Leaf, None)

    tmp_children_list = []
    len_tmp_children_list = 0
    for n in nodes:

        try:
            func = n.__class__._eval_func[-1]
            n_is_leaf = n.__class__._is_leaf
        except:
            if isinstance(n, Leaf) and leaf_func is not None:
                func = leaf_func
                n_is_leaf = True
            else:
                raise AssertionError("No lambda function associated with type: %s" % (n.__class__.__name__))

        if n_is_leaf:
            result = func(n, **args)
        else:
            len_children = len(n.children)
            if len_tmp_children_list < len_children:
                tmp_children_list.extend([None] * len_children)
                len_tmp_children_list = len(tmp_children_list)
            for i in range(len_children):
                ci = n.children[i]
                tmp_children_list[i] = all_results[ci]
            result = func(n, tmp_children_list[0:len_children], **args)

        all_results[n] = result

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results[node]


def eval_spn_top_down(root, eval_functions, all_results=None, parent_result=None, **args):
    """
    evaluates an spn top to down


    :param root: spnt root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, [parent_results], args**) and returns {child : intermediate_result}. This intermediate_result will be passed to child as parent_result. If intermediate_result is None, no further propagation occurs
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
    :param parent_result: initial input to the root node
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
    """
    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)

    all_results[root] = [parent_result]

    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            param = all_results[n]
            result = func(n, param, **args)

            if result is not None and not isinstance(n, Leaf):
                assert isinstance(result, dict)

                for child, param in result.items():
                    if child not in all_results:
                        all_results[child] = []
                    all_results[child].append(param)

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results[root]
