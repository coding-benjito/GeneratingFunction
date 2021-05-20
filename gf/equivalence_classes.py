# import sage.all
import numpy as np
import itertools
import string
import collections
import itertools
import operator
import functools
import copy
import sys
from timeit import default_timer as timer
import math
import string
from sympy import *
init_printing(use_unicode=True)
from string import ascii_lowercase

"""
Tree lingo:
 - bifurcating: every interior node has exactly 1 direct parent and 2 direct children
 - labeled: tips are uniquely labeled, we consider interior nodes to be unlabeled
 - unlabeled: tips are not labeled
 - rooted: there is a single root
 - unrooted: there is no single root
 - tree shape:  (Harding, 1971: unlabeled shape)
 - ranked/unranked: a ranked tree has ranked coalescence events
"""


def enumerate_bifurcating_labeled_rooted_trees_closed_form(tips):
    """
    We can easily derive that the result is 3 * 5 * 7 * ... * (2n-3), a product of odd integers.
    This is because teh n-th tip can be added to any of (2n-3) places.
    This is simple to see: Any addition yields 2 additional branches in the offspring tree.
    :param tips:
    :return: (2n-3)! / 2^(n-2) / (n-2)!
    Example: enumerate_bifurcating_labeled_rooted_trees_closed_form(3) --> 3
    """
    prod = 1
    for i in range(1, 2 * n - 3 + 1, 2):
        prod *= 2
    return int(prod)


def enumerate_bifurcating_labeled_unrooted_trees(tips):
    """
    An unrooted tree with n tips has 2n-3 rooted trees.
    :param tips:
    :return:
    """
    # Every rooted tree with (n-1) labeled tips corresponds to one unrooted tree.
    # Every unrooted tree with (n) labeled tips corresponds to one rooted tree with (n-1) labeled trees.
    # There are 1*3*5*...*(2n-5) unrooted bifurcating trees with n labeled tips
    return enumerate_bifurcating_labeled_rooted_trees_closed_form(tips - 1)


def enumerate_bifurcating_rooted_tree_shapes(tips):
    # Cavalli-Sforza and Edwards (1967) algorithm
    # Closed-form: Not known, but Donald Knuth (1973, p388) discusses a corresponding generating function.
    """
    The uniquely labeled tips have a unique ordering, e.g., (1, 2, 3) or ('a', 'b', 'c').
    We recursively build all bifurcating labeled rooted trees:
        Given a list of all possible bifurcating labeled rooted trees with n tips, for every element we add a single new
        tip on all possible internal branches and the 'root branch' to create all possible bifurcating labeled rooted
        trees with n+1 tips.
    :param tips: number of tips, a unique labeling and ordering is constructed here
    :return: dictionary that gives the number of bifurcating labeled rooted trees per number of tips (key) until tips
    enumerate_bifurcating_labeled_rooted_trees_closed_form(7) --> {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 6, 7: 11}
    """
    S = {
        1: 1
    }
    for i in range(2, tips + 1):
        S_i = 0
        if i % 2 == 1:
            for j in range(1, int(i / 2) + 1):
                S_i += S[j] * S[i - j]
        else:
            for j in range(1, round(i / 2 - 1) + 1):
                S_i += S[j] * S[i - j]
            S_i += S[round(i / 2)] * (S[round(i / 2)] + 1) / 2
        S[i] = round(S_i)
    return S


def construct_bifurcating_rooted_tree_shapes(tips):
    """
    for key integer you get the sample config, while for string you get the symmetric nodes
    the symmetric nodes are calculated at construction time

    Example:
        construct_bifurcating_rooted_tree_shapes(4)  -->
                        {1:   [['*']],
                         '1': [0],
                         2:   [[['*'], ['*']]],
                         '2': [1],
                         3:   [[['*'], [['*'], ['*']]]],
                         '3': [1],
                         4:   [[['*'], [['*'], [['*'], ['*']]]], [[['*'], ['*']], [['*'], ['*']]]],
                         '4': [1, 3]}
    :param tips:
    :return:
    """
    # '*' is n=1
    # '(*,*)' is n=2
    # '(*,(*,*))' is n=3, etc.

    tree_shapes = {
        1: [['*']],
        '1': [0]
    }
    for i in range(2, tips + 1):
        temp = list()
        sym_splits = list()

        if i % 2 == 1:
            for j in range(1, int(i / 2) + 1):
                for k in range(len(tree_shapes[j])):
                    for l in range(len(tree_shapes[i - j])):
                        temp.append([tree_shapes[j][k],
                                     tree_shapes[i - j][l]])
                        sym_splits.append(tree_shapes[str(j)][k] +
                                          tree_shapes[str(i - j)][l])

        else:
            for j in range(1, round(i / 2)):
                for k in range(len(tree_shapes[j])):
                    for l in range(len(tree_shapes[i - j])):
                        temp.append([tree_shapes[j][k],
                                     tree_shapes[i - j][l]])
                        sym_splits.append(tree_shapes[str(j)][k] +
                                          tree_shapes[str(i - j)][l])
            j = round(i / 2)
            for k in range(len(tree_shapes[j])):
                for l in range(k, len(tree_shapes[i - j])):
                    temp.append([tree_shapes[j][k],
                                 tree_shapes[i - j][l]])
                    if k == l:
                        sym_splits.append(tree_shapes[str(j)][k] +
                                          tree_shapes[str(i - j)][l] + 1)
                    else:
                        sym_splits.append(tree_shapes[str(j)][k] +
                                          tree_shapes[str(i - j)][l])
        tree_shapes[i] = temp
        tree_shapes[str(i)] = sym_splits


def fish_representative_from_bifurcating_rooted_tree_shapes(n, tree_shapes):
    """
    we know that gf yield constant solution across items of equivalence class
    hence, one representative has to be evaluated and for this unique labels are required
    here we just give a unique label to the wildcards
    :param tips:
    :return: a list of representatives for every equivalence class
    """
    representatives = list()
    for i in range(len(tree_shapes[n])):
        to_fill = str(tree_shapes[n][i])
        for j in range(n):
            to_fill = to_fill.replace('*', string.ascii_lowercase[j], 1)
        representatives.append(eval(to_fill))
    return representatives


def give_n_h(n_s, n):
    """

    Example:
        give_n_h(2, 3) --> 2
    :param n_s:
    :param n:
    :return:
    """
    return round(math.factorial(n) / 2 ** n_s)
	# for multiple populations: 1/2 ** n_s * prod(factorial(n_i))


def give_equiv_class_sizes(n, tree_shapes):
    """
    Returns the number of contained bifurcating labeled unrooted trees within all equivalence classes with n tips.
    in same order as the representatives come along

    Example:
    give_equiv_class_sizes(6) --> [360, 90, 180, 180, 45, 90]

    :param n:
    :return:
    """
    equiv_class_sizes = list()
    for n_s in tree_shapes[str(n)]:
        equiv_class_sizes.append(give_n_h(n_s=n_s, n=n))
    return equiv_class_sizes


"""
Partitioning the GF into equivalence classes (Lohse et al., 2011, 2016)
To condition on a certain topology the incompatible terms in the GF are set to 0.
At the same time the total summed rate across all lambda is unaffected by that.
If one sets the omegas within the topology-conditioned GF to 0, the probability of that topology is returned.
Each equivalence class defines a set of identically distributed genealogy.

"""


def lookup_equivalence_class_membership()
	pass

def give_psi(n, method='partitioning'):
	tree_shapes = construct_bifurcating_rooted_tree_shapes(n)
	representatives = fish_representative_from_bifurcating_rooted_tree_shapes(n, tree_shapes)
	n_h = give_equiv_class_sizes(n, tree_shapes)
	assert len(representatives) == len(n_h), 'unequal lengths'
	return sum([n_h[i] * give_psi(representatives[i] --- having branches from here --- ) for i in range(representatives)])


def give_extant_samples(n):
    return ascii_lowercase[0:n]

def give_combos(alphabet):
    l = len(alphabet)
    for k in range(1, l+1):
        give_combo(alphabet, '', l, k)

def give_combo(alphabet, prefix, l, k, i_0=0):
if k==0:
combos.append(prefix)
else:
for i in range(i_0, l):
new_prefix = prefix + alphabet[i]
give_combo(alphabet, new_prefix, l, k-1, i+1)

"""
def binary_splits():
    pass
"""
