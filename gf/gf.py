#!/usr/bin/env python3

"""
Generating function (GF)
"""

import collections
import copy
import functools
import itertools
import numpy as np
import operator
import sage.all
import string
import sys
from timeit import default_timer as timer
from sympy import *


# auxiliary functions
def powerset(iterable):
    """
	Creates the power set of the provided set with string items. Elements of the subsets are concatenated.
	Example:
		list(powerset(['a', 'b', 'c'])) --> ['', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc']
	Args:
		iterable: an iterable (e.g., list) with string items
	Returns: generator containing all possible 2^len(iterable) subsets (concatenated and ordered) of iterable
	"""
    s = list(iterable)
    return (''.join(sorted(subelement)) for subelement in (itertools.chain.from_iterable(itertools.combinations(s, r)
                                                                                         for r in range(len(s) + 1))))


def flatten(iterable):
    """
	Flattens an iterable from depth n to depth (n-1).
	Examples:
		list(flatten(['a', 'b', 'c'])) --> ['a', 'b', 'c']
		list(flatten([['a', 'b', 'c']])) --> ['a', 'b', 'c']
		list(flatten([[['a']], 'b', ['c']])) --> [['a'], 'b', 'c']
	Args:
		iterable: an iterable (e.g., list) with arbitrary items (e.g., nested lists)
	Returns: iterator where the second layer of depth was removed if existent
	"""
    return itertools.chain.from_iterable(iterable)


def coalesce_lineages(config, to_join):
    """
	Joins lineages and provides the resulting tuple of newly created lineages for a single population.
	Example:
		coalesce_lineages(['a', 'b', 'c'], ['a']) --> ['a', 'b', 'c']
		coalesce_lineages(['a', 'b', 'c'], ['a', 'b']) --> ['ab', 'c']
	Args:
		config: sample configuration which contains all the available lineages (iterable with strings)
		to_join: listing of all lineages to be joined (iterable with strings)
	Returns: tuple with config of lineages after coalescence has occured
	"""
    result = list(config)
    for lineage in to_join:
        result.remove(lineage)
    result.append(''.join(sorted(flatten(to_join))))
    result.sort()
    return tuple(result)


def coalesce_single_pop(config, coal_rate=1.):
    """
	Generates all possible (pairwise) coalescence events for a single population.
	Example:
		list(coalesce_single_pop(['a', 'b', 'c'])) --> [(1.0, ('ab', 'c')), (1.0, ('ac', 'b')), (1.0, ('a', 'bc'))]
		list(coalesce_single_pop(['ab', 'c'])) --> [(1.0, ('abc',))]
	Args:
		config: iterable containing all lineages as strings present within the population
		coal_rate: a sage_var (float) with the rate at which coalescence happens in that population, scaling
	Returns: generator with all possible coalescence events and associated rates
	"""
    coal_event_pairs = list(itertools.combinations(config, 2))
    coal_counts = collections.Counter(coal_event_pairs)  # can be omitted as it associates 1 with every element
    for lineages, count in coal_counts.items():
        result = coalesce_lineages(config, lineages)
        yield count * coal_rate, result


# dealing with mutation/branchtypes
def make_branchtype_dict(sample_list, mapping='unrooted', labels=None):
    """
	Maps lineages to their respective mutation types.

	Examples:
		make_branchtype_dict([('a', 'a', 'b', 'b')], mapping='unrooted') -->
				{'a': m_2,   # het. at a
				'b': m_1,    # het. at b
				'aa': m_4,   # fixed diff.
				'ab': m_3,   # het. at a and b
				'bb': m_4,   # fixed diff.
				'aab': m_1,  # het. at b
				'abb': m_2}  # het. at a
		make_branchtype_dict([('a', 'a', 'b', 'b')], mapping='label') --> {'a': z_a, 'b': z_b, ..., 'abb': z_abb}

	Args:
		sample_list:
		mapping:
		labels:

	Returns:

	"""
    all_branchtypes = list(flatten(sample_list))
    branches = [branchtype for branchtype in powerset(all_branchtypes) if 0 < len(branchtype) < len(all_branchtypes)]
    if mapping.startswith('label'):
        if labels:
            assert len(branches) == len(labels), "number of labels does not match number of branchtypes"
            branchtype_dict = {branchtype: sage.all.SR.var(label) for branchtype, label in zip(branches, labels)}
        else:
            branchtype_dict = {branchtype: sage.all.SR.var(f'z_{branchtype}') for branchtype in branches}
    elif mapping == 'unrooted':  # this needs to be extended to the general thing!
        if not labels:
            labels = ['m_1', 'm_2', 'm_3', 'm_4']
        assert set(all_branchtypes) == {'a', 'b'}
        branchtype_dict = dict()
        for branchtype in powerset(all_branchtypes):
            if branchtype in ('abb', 'a'):
                branchtype_dict[branchtype] = sage.all.SR.var(labels[1])  # hetA
            elif branchtype in ('aab', 'b'):
                branchtype_dict[branchtype] = sage.all.SR.var(labels[0])  # hetB
            elif branchtype == 'ab':
                branchtype_dict[branchtype] = sage.all.SR.var(labels[2])  # hetAB
            else:
                branchtype_dict[branchtype] = sage.all.SR.var(labels[3])  # fixed difference
    else:
        ValueError("This branchtype mapping has not been implemented yet.")
    return branchtype_dict


def sort_mutation_types(branchtypes):
    if isinstance(branchtypes, dict):
        return sorted(set(branchtypes.values()), key=lambda x: str(x))
    elif isinstance(branchtypes, list) or isinstance(branchtypes, tuple):
        return sorted(set(branchtypes), key=lambda x: str(x))
    else:
        raise ValueError(f'sort_mutation_types not implemented for {type(branchtypes)}')


# processing generating function
def inverse_laplace(equation, dummy_variable):
    return (sage.all.inverse_laplace(subequation / dummy_variable, dummy_variable, sage.all.SR.var('T', domain='real'),
                                     algorithm='giac') for subequation in equation)


def split_gf(gf, chunksize):
    # splitting gf generator using chunksize
    i = iter(gf)
    piece = sum(itertools.islice(i, chunksize))
    while piece:
        yield piece
        piece = sum(itertools.islice(i, chunksize))


def split_gf_iterable(gf, chunksize):
    # splitting gf generator using chunksize
    i = iter(gf)
    piece = list(itertools.islice(i, chunksize))
    while piece:
        yield piece
        piece = list(itertools.islice(i, chunksize))


class GFObject:
    """
	The generating function (GF) class.
	"""

    def __init__(self, sample_list, coalescence_rates, branchtype_dict, migration_direction=None, migration_rate=None,
                 exodus_direction=None, exodus_rate=None):
        """
		Constructor

		Examples:
			Base:
			sample_list = [('a', 'a', 'b', 'b')]
			branchtype_dict = make_branchtype_dict(sample_list, mapping='unrooted')
			GFObject(sample_list, (1, ), branchtype_dict)

			Migr: GFObject(sample_list, (1., 1., 1.), branchtype_dict, migration_direction=[(1, 2)], migration_rate=1)

			Ex.:  GFObject(sample_list, (1., 1., 1.), branchtype_dict, exodus_direction=[(1, 2, 0), exodus_rate=1.)

		Args:
			sample_list: list with tuples (sub-populations) with strings (lineages)
			coalescence_rates: list with floats giving the coalescence rates in the sub-populations
			branchtype_dict: gives a name to every branch lying directly parental to certain lineages
			migration_direction: list with tuples with 2 integers (indices),
								 for n sub-populations there are n(n-1)=2*n! possible directions
			migration_rate: rate associated with any migration direction
			exodus_direction: analogously to migration direction (source population is left empty)
			exodus_rate: rate associated with any exodus direction
		"""
        assert len(sample_list) == len(coalescence_rates)
        if sum(1 for pop in sample_list if len(pop) > 0) > 1:
            assert migration_direction or exodus_direction, 'lineages from different populations cannot coalesce without migration or exodus event.'
        self.sample_list = tuple(tuple(sorted(pop)) for pop in sample_list)
        self.coalescence_rates = coalescence_rates
        self.branchtype_dict = branchtype_dict
        self.migration_direction = migration_direction
        if migration_direction and not migration_rate:
            self.migration_rate = sage.all.SR.var('M')  # set domain???
        else:
            self.migration_rate = migration_rate
        self.exodus_direction = exodus_direction
        if exodus_direction and not exodus_rate:
            self.exodus_rate = sage.all.SR.var('E')  # set domain?
        else:
            self.exodus_rate = exodus_rate

    def coalescence_events(self, state_list):
        """
		Determines new population configurations and associated rates due to a COALESCENCE_EVENT.

		Example:
			[1]
			self.coalescence_rates --> (1, )
			self.coalescence_events([('a', 'b'), ('c', 'd')]) --> [(1.0, (('ab',), ('c', 'd')))]

			[2]
			self.coalescence_rates --> (1, 1)  # coalescent rate per sub-population
			self.coalescence_events([('a', 'b'), ('c', 'd')]) --> [(1.0, (('ab',), ('c', 'd'))),
																   (1.0, (('a', 'b'), ('cd',)))]

		Args:
			state_list: a list of population tuples containing lineages (str)

		Returns: Generator yielding all possible new population configurations and rates due to coalescence

		"""
        result = []
        for idx, (pop, rate) in enumerate(zip(state_list, self.coalescence_rates)):
            for count, coal_event in coalesce_single_pop(pop):
                modified_state_list = list(state_list)
                modified_state_list[idx] = coal_event
                result.append((count * rate, tuple(modified_state_list)))
        # yield((count, tuple(modified_state_list)))
        return result

    def migration_events(self, state_list):
        """
		Determines new population configurations and associated rates due to MIGRATION_EVENT (backwards in time).

		Examples:
			self.sample_list = [('a', 'b'), ('c', 'd'), ('e', 'f')]
			self.coalescence_rates = (1, 1, 1)
			self.migration_direction = [(1, 2)]
			self.migration_rate = 1
			self.migration_events(self.sample_list) --> [(1, (('a', 'b'), ('d',), ('c', 'e', 'f'))),
														 (1, (('a', 'b'), ('c',), ('d', 'e', 'f')))]

		Args:
			state_list: a list of population tuples containing lineages (str)
		Returns: Generator yielding all possible new population configurations and rates due to migration
		"""
        result = []
        if self.migration_direction:
            for source, destination in self.migration_direction:
                lineage_count = collections.Counter(state_list[source])
                for lineage, count in lineage_count.items():
                    temp = list(state_list)
                    idx = temp[source].index(lineage)
                    temp[source] = tuple(temp[source][:idx] + temp[source][idx + 1:])
                    temp[destination] = tuple(sorted(list(temp[destination]) + [lineage, ]))
                    result.append((count * self.migration_rate, tuple(temp)))
        return result

    def exodus_events(self, state_list):
        """
		Determines new population configurations and associated rates due to EXODUS_EVENT (backwards in time).

		Examples:
			self.sample_list = [('a', 'b'), ('c', 'd'), ('e', 'f')]
			self.coalescence_rates = (1, 1, 1)
			self.exodus_direction = [(1, 2)]
			self.exodus_rate = 1
			self.exodus_events(self.sample_list) --> [(1, (('a', 'b'), (), ('c', 'd', 'e', 'f')))]

		Args:
			state_list: a list of population tuples containing lineages (str)
		Returns: Generator yielding all possible new population configurations and rates due to exoduses
		"""
        result = []
        if self.exodus_direction:
            for source, destination in self.exodus_direction:
                temp = list(state_list)
                sources_joined = tuple(itertools.chain.from_iterable([state_list[idx] for idx in source]))
                if len(sources_joined) > 0:
                    temp[destination] = tuple(sorted(state_list[destination] + sources_joined))
                    for idx in source:
                        temp[idx] = ()
                    result.append((self.exodus_rate, tuple(temp)))
        return result

    def rates_and_events(self, state_list):
        """
		Returns all possible events and respective rates.

		Example:
			See sub-routines.

		Args:
			state_list: a list of population tuples containing lineages (str)
		Returns: joined list of COALESCENCE_EVENTs, MIGRATION_EVENTs and EXODUS_EVENTs
		"""
        c = self.coalescence_events(state_list)
        m = self.migration_events(state_list)
        e = self.exodus_events(state_list)
        return c + m + e

    def gf_single_step(self, gf_old, state_list):
		"""
		Returns the result of a single recursion (single tail recursion step) for the GF.
		Args:
			gf_old: SAGE expression which is a result form the previous recursion step
			state_list: list of sub-population tuples containing lineages with strings

		Returns: list with GFs up until all considered events in outcomes
		"""
        current_branches = list(flatten(state_list))  # in its typical behavior the demes are collected together
        numLineages = len(current_branches)
        if numLineages == 1:
            ValueError('gf_single_step fed with single lineage, should have been caught.')
        else:
            outcomes = self.rates_and_events(state_list)
            total_rate = sum([rate for rate, state in outcomes])
            dummy_sum = sum(self.branchtype_dict[b] for b in current_branches)
            # for rate, new_state_list in outcomes:
            #	yield (gf_old*rate*1/(total_rate + dummy_sum), new_state_list)
            return [(gf_old * rate * 1 / (total_rate + dummy_sum), new_state_list) for rate, new_state_list in outcomes]

    def make_gf(self):
		"""
		[generator returning generating function]
		Yields:
			[type] -- []
		Args:
			self:

		Returns:

		"""
        stack = [(1, self.sample_list)]
        result = []
        while stack:
            gf_n, state_list = stack.pop()
            if sum(len(pop) for pop in state_list) == 1:
                yield gf_n
            else:
                for gf_nplus1 in self.gf_single_step(gf_n, state_list):
                    stack.append(gf_nplus1)

    def gf_single_step_graph(self, tracking_list, state_list):
        current_branches = list(flatten(state_list))
        numLineages = len(current_branches)
        if numLineages == 1:
            ValueError('gf_single_step fed with single lineage, should have been caught.')
        else:
            outcomes = self.rates_and_events(state_list)
            for rate, new_state_list in outcomes:
                yield (tracking_list[:] + [(rate, new_state_list), ], new_state_list)

    def make_gf_graph(self):
        """[generator returning gf graph]
		Yields:
			[type] -- []
		"""
        stack = [([(1, tuple(self.sample_list))], self.sample_list)]
        result = []
        while stack:
            tracking_list, state_list = stack.pop()
            if sum(len(pop) for pop in state_list) == 1:
                yield tracking_list
            else:
                for new_step in self.gf_single_step_graph(tracking_list, state_list):
                    stack.append(new_step)
