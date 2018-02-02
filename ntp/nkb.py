# coding=utf-8
"""
   _  ____ _____
  / |/ / //_/ _ )
 /    / ,< / _  |
/_/|_/_/|_/____/ v0.6

Neural Knowledge Base
"""

import tensorflow as tf

from ntp.kb import Atom
from pprint import pprint
from ntp.util import is_variable, is_parameter
from ntp.jtr.preprocess.vocab import Vocab
from ntp.jtr.util.tfutil import unit_length_transform
import numpy as np
import copy


def rule2struct(rule):
    """
    Returns the structure of a rule used to partition a knowledge base
    :param rule: a rule
    :return: a tuple representing the structure of the rule
    """
    predicates = {}
    constants = {}
    variables = {}
    struct = []
    for predicate, args in rule:
        atom_struct = []
        if predicate not in predicates:
            predicates[predicate] = "p" + str(len(predicates))
        atom_struct.append(predicates[predicate])
        for arg in args:
            if is_variable(arg):
                if arg not in variables:
                    variables[arg] = "X" + str(len(variables))
                atom_struct.append(variables[arg])
            else:
                if arg not in constants:
                    constants[arg] = "c" # + str(len(constants))
                atom_struct.append(constants[arg])
        struct.append(tuple(atom_struct))
    return tuple(struct)


def augment_with_templates(kb, rule_templates):
    """
    :param kb: a knowledge base with symbolic representations
    :return: knowledge base agumented with parameterized rule templates
    """
    kb_copy = copy.deepcopy(kb)

    def suffix_rule_parameters(rule, num_rule, num_copy):
        new_rule = []
        for predicate, args in rule:
            if is_parameter(predicate):
                new_rule.append(Atom("%s_%d_%d" %
                                     (predicate, num_rule, num_copy), args))
            else:
                new_rule.append(Atom(predicate, args))
        return new_rule

    for i, (rule_template, num) in enumerate(rule_templates):
        for j in range(num):
            # fixme: need to suffix parameters by i and j
            kb_copy.append(suffix_rule_parameters(rule_template, i, j))
    return kb_copy


def partition(kb):
    """
    :param kb: a knowledge base with symbolic representations
    :return: a knowledge base partitioned by the structure of rules
    """
    kb_partitioned = {}
    for rule in kb:
        struct = rule2struct(rule)
        if struct not in kb_partitioned:
            kb_partitioned[struct] = [rule]
        else:
            kb_partitioned[struct].append(rule)
    return kb_partitioned


def kb2ids(kb, vocab=None, permutation=[0, 1, 2, 3]):
    """
    :param kb: a partitioned knowledge base
    :return: a partitioned knowledge base where symbols (except variables) are
    mapped to ids
    """
    kb_ids = {}
    vocab = vocab or Vocab()

    predicate_ids = []
    constant_ids = []

    keys = sorted(list(kb.keys()))
    print("before")
    pprint(keys)    # permutation = [1, 3, 0, 2]

    if permutation:
        keys = [x for i, x in sorted(zip(permutation, keys))]

    print("after")
    # fixme: non-deterministic!
    for struct in keys:
        print(struct)
        rules = kb[struct]
        kb_stacked = []

        for rule in rules:
            for i, (predicate, args) in enumerate(rule):
                if len(kb_stacked) < i + 1:
                    kb_stacked.append([])
                symbols = [x for x in [predicate] + args]  # if not is_variable(x)]
                for j, sym in enumerate(symbols):
                    if not is_variable(sym):
                        if j == 0 and sym not in vocab:
                            predicate_ids.append(vocab(sym))
                        elif j > 0 and sym not in vocab:
                            constant_ids.append(vocab(sym))

                    if len(kb_stacked[i]) < j + 1:
                        kb_stacked[i].append([])
                    kb_stacked[i][j].append(sym)

        # mapping to ids and stacking as numpy array
        for i, atom in enumerate(kb_stacked):
            for j, symbols in enumerate(atom):
                if not is_variable(symbols[0]):
                    kb_stacked[i][j] = np.hstack(vocab(symbols))
                else:
                    kb_stacked[i][j] = symbols

        kb_ids[struct] = kb_stacked
    return kb_ids, vocab, predicate_ids, constant_ids


def embed_symbol(symbol, embedding_matrix, unit_normalize=True, dim=1,
                 keep_prob=1.0):
    symbol_embedded = tf.nn.embedding_lookup(embedding_matrix, symbol)
    if unit_normalize:
        symbol_embedded = unit_length_transform(symbol_embedded, dim)

    if keep_prob != 1.0:
        symbol_embedded = tf.nn.dropout(symbol_embedded, keep_prob)

    return symbol_embedded


def kb2nkb(kb, input_size=10, vocab=None, unit_normalize=True,
           init=(-1.0, 1.0), keep_prob=1.0, emb = None, permutation=None):
    """
    Embeds symbols in a kb
    :param kb: a knowledge base with symbolic representations
    :return: a partitioned knowledge base with trainable vector representations
    """

    kb_partitioned = partition(kb)

    kb_ids, vocab, predicate_ids, constant_ids = kb2ids(kb_partitioned, vocab, permutation)

    # with tf.variable_scope("nkb", reuse=None) as scope:
    initializer = tf.contrib.layers.xavier_initializer()
    if init is not None:
        initializer = tf.random_uniform_initializer(init[0], init[1])


    if emb is None:
        embedding_matrix = \
            tf.get_variable(
                "embeddings", [len(vocab), input_size],
                # initializer=tf.random_uniform_initializer(-1.0, 1.0)
                initializer=initializer
                #tf.random_normal_initializer()
            )
    else:
        embedding_matrix = emb

    nkb = {}
    for ix, struct in enumerate(kb_ids):
        kb_stacked = kb_ids[struct]

        atoms_embedded = []
        for i, atom in enumerate(kb_stacked):
            atom_embedded = []
            for j in range(len(kb_stacked[i])):
                symbol = kb_stacked[i][j]
                if isinstance(symbol, np.ndarray):
                    atom_embedded.append(
                        embed_symbol(symbol, embedding_matrix,
                                     unit_normalize=unit_normalize,
                                     keep_prob=keep_prob))
                else:
                    if isinstance(symbol, list):
                        atom_embedded.append("%s%d" % (symbol[0][0], ix))
                    # atom_embedded.append(symbol)
            atoms_embedded.append(atom_embedded)
        nkb[struct] = atoms_embedded

    return nkb, kb_ids, vocab, embedding_matrix, predicate_ids, constant_ids
