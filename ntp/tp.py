"""
 _________
/_  __/ _ \
 / / / ___/
/_/ /_/ v1.0

Theorem Provers based on Backward Chaining

http://stackoverflow.com/questions/33857541/backward-chaining-algorithm
https://github.com/gnufs/aima-java/blob/master/aima-core/src/main/java/aima/core/logic/fol/Unifier.java
"""

import copy
from ntp.kb import Atom, load_from_file
from pprint import pprint
import collections
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import *
import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import sys

FAILURE = "FAILURE"
SUCCESS = "SUCCESS"


class LazyWord2Vec(object):
    def __init__(self, path):
        self.path = path
        self.word2vec = None

    def get_model(self):
        if self.word2vec is None:
            print("Loading word2vec...")
            self.word2vec = Word2Vec.load_word2vec_format(self.path,binary=True)
            print("Done!")
        return self.word2vec

lazy_word2vec = LazyWord2Vec('path-to/GoogleNews-vectors-negative300.bin')


def is_atom(arg):
    return isinstance(arg, Atom)


def is_variable(arg):
    if isinstance(arg, str):
        return arg.isupper()
    else:
        return False


def atom2string(atom):
    return "%s(%s)" % (atom.predicate, ",".join(atom.arguments))


def rule2string(rule):
    head = atom2string(rule[0])
    body = [atom2string(x) for x in rule[1:]]
    if len(rule) == 1:
        return "%s." % head
    else:
        return "%s :- %s." % (head, ", ".join(body))


def subs2string(substitutions):
    return "{%s}" % ", ".join([key+"/"+val for
                               key, val in substitutions.items()])


def is_ground_atom(atom):
    if is_atom(atom):
        return len([x for x in atom.arguments if is_variable(x)]) == 0
    else:
        return False


def has_free_variables(rule):
    return len([atom for atom in rule if not is_ground_atom(atom)]) > 0


def normalize(kb):
    counter = 0
    normalized_kb = []

    def suffix_variables(atom, suffix):
        new_args = []
        for arg in atom.arguments:
            if is_variable(arg):
                new_args.append(arg+suffix)
            else:
                new_args.append(arg)
        return Atom(atom.predicate, new_args)

    for rule in kb:
        if has_free_variables(rule):
            normalized_kb.append([suffix_variables(atom, str(counter))
                                  for atom in rule])
            counter += 1
        else:
            normalized_kb.append(rule)
    return normalized_kb


def unify_variable(variable, x, substitutions, depth=0):
    #print(" " * depth * 4 + str(substitutions))
    substitutions[variable] = x
    return substitutions


def sentence2vecs(sentence, delimiter="\s|\-|\""):
    words = [x for x in re.compile(delimiter).split(sentence)
             if x != "" and x not in ENGLISH_STOP_WORDS]
    word2vec = lazy_word2vec.get_model()
    vecs = []
    for word in words:
        if word in word2vec:
            vecs.append(word2vec[word])
    return vecs


def check_for_equality(rhs, goal, symbolic=False, threshold=1-1e9,
                       aggregate=np.sum, min_content_words=2):
    if symbolic:
        return rhs == goal
    else:
        if isinstance(rhs, str) and isinstance(goal, str) \
                and not is_variable(rhs) and not is_variable(goal):
            if rhs == goal:
                return True
            else:
                # check whether similar in vector space
                rhs_vecs = sentence2vecs(rhs)
                goal_vecs = sentence2vecs(goal)
                if len(rhs_vecs) >= min_content_words \
                        and len(goal_vecs) >= min_content_words:
                    rhs_rep = aggregate(rhs_vecs)/len(rhs_vecs)
                    goal_rep = aggregate(goal_vecs)/len(goal_vecs)
                    # fixme: should be between 0 and 1, but can get above 1
                    #sim = 1 - cosine(rhs_rep, goal_rep)
                    #sim = 1 - sqeuclidean(rhs_rep, goal_rep)
                    sim = 1 - euclidean(rhs_rep, goal_rep)
                    # print(rhs, len(rhs_vecs), goal, len(goal_vecs), sim)
                    if sim > threshold:
                        return True
                return False
        else:
            return rhs == goal


def unify(rhs, goal, substitutions, depth=0, symbolic=True):
    substitutions_copy = copy.deepcopy(substitutions)
    if substitutions_copy == FAILURE:
        return substitutions_copy
    elif check_for_equality(rhs, goal, symbolic):
        return substitutions_copy
    elif is_variable(rhs):
        return unify_variable(rhs, goal, substitutions_copy, depth)
    elif is_variable(goal):
        return unify_variable(goal, rhs, substitutions_copy, depth)
    elif is_atom(rhs) and is_atom(goal):
        return unify(rhs.arguments, goal.arguments,
                     unify(rhs.predicate, goal.predicate, substitutions_copy,
                           depth, symbolic),
                     depth, symbolic)
    elif isinstance(rhs, list) and isinstance(goal, list) \
            and len(rhs) == len(goal):
        return unify(rhs[0], goal[0],
                     unify(rhs[1:], goal[1:], substitutions_copy,
                           depth, symbolic),
                     depth, symbolic)
    else:
        return FAILURE


def substitute(goal, substitutions, depth=0):
    for i, arg in enumerate(goal.arguments):
        if is_variable(arg) and arg in substitutions:
            goal.arguments[i] = substitutions[arg]
    return goal


def flatten_proofs(proofs):
    def flatten(xs):
        for x in xs:
            if isinstance(x, collections.Iterable) \
                    and not isinstance(x, str) \
                    and not isinstance(x, dict):
                for sub in flatten(x):
                    yield sub
            else:
                yield x

    return list(flatten(proofs))


def or_(kb, goal, substitutions=dict(), depth=0, trace=False, symbolic=True):
    """
    :param kb: A list of rules, which is itself a list of atoms.
    :param goal: An atom to prove.
    :param substitutions: The upstream substitutions, initially empty.
    :param depth: Depth of the prover.
    :return: List of downstream substitutions.
    """
    proofs = []
    for rule in kb:
        head = rule[0]
        body = rule[1:]
        # print("head ["+str(head)+"] body ["+str(body)+"]")
        substitutions_ = unify(head, goal, substitutions, depth, symbolic)
        if substitutions_ != FAILURE:
            if trace:
                print(" " * (4 * depth) + "Rule: " + rule2string(rule))
                print(" " * (4 * depth + 4) + subs2string(substitutions_))
            proof = and_(kb, body, substitutions_, depth, trace, symbolic)
            proofs.append(proof)
    return flatten_proofs(proofs)


def and_(kb, subgoals, substitutions, depth=0, trace=False, symbolic=True):
    """
    :param kb: A list of rules, which is itself a list of atoms.
    :param subgoals: A list of atoms to prove.
    :param substitutions: The upstream substitutions.
    :param depth: Depth of the prover.
    :return: Downstream substitutions.
    """
    if len(subgoals) == 0:
        return substitutions
    elif not symbolic and depth > 2:
        return FAILURE
    else:
        head = subgoals[0]
        body = subgoals[1:]
        goal = substitute(head, substitutions)
        if trace:
            print(" " * (4 * depth + 4) + atom2string(goal) + "?")
        proofs = []
        for substitutions_ in or_(kb, goal, substitutions, depth+1, trace, symbolic):
            proofs.append(and_(kb, body, substitutions_, depth, trace, symbolic))
        return proofs


def get_free_variables(goal):
    return [x for x in goal.arguments if is_variable(x)]


def prove(kb, goal, symbolic=True, trace=False):
    substitutions = or_(normalize(kb), goal, trace=trace, symbolic=symbolic)
    free_variables = get_free_variables(goal)
    if len(free_variables) > 0:
        if len(substitutions) == 0:
            return FAILURE
        else:
            answers = []
            for substitution in substitutions:
                answer = dict()
                # fixme: something wrong here
                if substitution != FAILURE:
                    for x in free_variables:
                        answer[x] = substitution[x]
                    answers.append(answer)
            return answers
    else:
        if len(substitutions) == 0:
            return FAILURE
        else:
            return SUCCESS


if __name__ == '__main__':
    kb = load_from_file("./data/ntp/simpsons.nl")
    goal = Atom('grandchildOf', ["Q", "abe"])
    result = prove(kb, goal, trace=True)
    pprint(result)

