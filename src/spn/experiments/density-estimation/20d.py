from joblib import Memory
from matplotlib.colors import LogNorm, PowerNorm
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learnspn_b
from numpy import genfromtxt
import numpy as np
from spn.algorithms.Marginalization import marginalize
from spn.structure.Base import Context, Sum
from spn.structure.StatisticalTypes import MetaType
from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import Bernoulli, Categorical
from spn.structure.leaves.parametric.Inference import add_parametric_inference_support
import matplotlib.cm as cm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from spn.data.datasets import get_binary_data, get_nips_data
from spn.algorithms.Statistics import get_structure_stats

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


import os
import csv


DEBD = ['accidents',
        'ad',
        'baudio',
        'bbc',
        'bnetflix',
        'book',
        'c20ng',
        'cr52',
        'cwebkb',
        'dna',
        'jester',
        'kdd',
        'kosarek',
        'moviereview',
        'msnbc',
        'msweb',
        'nltcs',
        'plants',
        'pumsb_star',
        'tmovie',
        'tretail',
        'voting']

DEBD_num_vars = {
    'accidents': 111,
    'ad': 1556,
    'baudio': 100,
    'bbc': 1058,
    'bnetflix': 100,
    'book': 500,
    'c20ng': 910,
    'cr52': 889,
    'cwebkb': 839,
    'dna': 180,
    'jester': 100,
    'kdd': 64,
    'kosarek': 190,
    'moviereview': 1001,
    'msnbc': 17,
    'msweb': 294,
    'nltcs': 16,
    'plants': 69,
    'pumsb_star': 163,
    'tmovie': 500,
    'tretail': 135,
    'voting': 1359}

DEBD_display_name = {
    'accidents': 'accidents',
    'ad': 'ad',
    'baudio': 'audio',
    'bbc': 'bbc',
    'bnetflix': 'netflix',
    'book': 'book',
    'c20ng': '20ng',
    'cr52': 'reuters-52',
    'cwebkb': 'web-kb',
    'dna': 'dna',
    'jester': 'jester',
    'kdd': 'kdd-2k',
    'kosarek': 'kosarek',
    'moviereview': 'moviereview',
    'msnbc': 'msnbc',
    'msweb': 'msweb',
    'nltcs': 'nltcs',
    'plants': 'plants',
    'pumsb_star': 'pumsb-star',
    'tmovie': 'each-movie',
    'tretail': 'retail',
    'voting': 'voting'}


def load_debd(data_dir, name, dtype='int32'):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    train_path = os.path.join(data_dir,  name + '.ts.data')
    test_path = os.path.join(data_dir,  name + '.test.data')
    valid_path = os.path.join(data_dir, name + '.valid.data')

    reader = csv.reader(open(train_path, 'r'), delimiter=',')
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, 'r'), delimiter=',')
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, 'r'), delimiter=',')
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x


def get_binary_data(name, path='spn/data/'):
    train = np.loadtxt(path + "/binary/" + name + ".ts.data",
                       dtype=float, delimiter=",", skiprows=0)
    test = np.loadtxt(path + "/binary/" + name + ".test.data",
                      dtype=float, delimiter=",", skiprows=0)
    valid = np.loadtxt(path + "/binary/" + name + ".valid.data",
                       dtype=float, delimiter=",", skiprows=0)
    D = np.vstack((train, test, valid))
    F = D.shape[1]
    features = ["V" + str(i) for i in range(F)]

    return name.upper(), np.asarray(features), D, train, valid, test, np.asarray(["discrete"] * F), np.asarray(["bernoulli"] * F)


add_parametric_inference_support()
# memory = Memory(cachedir="cache", verbose=0, compress=9)

train, valid, test = load_debd('spn/data/binary', 'accidents')

n_features = train.shape[1]
types = [MetaType.DISCRETE for i in range(n_features)]

ds_context = Context(meta_types=types)
ds_context.parametric_types = [Categorical for i in range(n_features)]
ds_context.statistical_type = types
ds_context.add_domains(train)

print("train data shape", train.shape)
spn = learnspn_b(train, ds_context, min_instances_slice=200, threshold=5, memory=None)

print(get_structure_stats(spn))
train_ll = log_likelihood(spn, train)
print('train ll', train.mean())
valid_ll = log_likelihood(spn, valid)
print('valid ll', valid.mean())
test_ll = log_likelihood(spn, test)
print('test ll', test.mean())
