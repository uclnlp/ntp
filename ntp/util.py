# coding=utf-8
"""
         __  _ __
  __  __/ /_(_) /
 / / / / __/ / /
/ /_/ / /_/ / /
\__,_/\__/_/_/ v0.2

Making useful stuff happen since 2016
"""

import re
import collections
import numpy as np

Atom = collections.namedtuple("Atom", ["predicate", "arguments"])


def trim(string):
    """
    :param string: an input string
    :return: the string without trailing whitespaces
    """
    return re.sub("\A\s+|\s+\Z", "", string)


def is_atom(arg):
    return isinstance(arg, Atom)


def is_variable(arg):
    if isinstance(arg, str):
        return arg.isupper()
    else:
        return False


def is_list(arg):
    return isinstance(arg, list)


def is_array(arg):
    return isinstance(arg, np.ndarray)


def is_constant(arg):
    if isinstance(arg, str):
        return arg.islower()
    else:
        return False


def is_parameter(predicate):
    if isinstance(predicate, str):
        return predicate[0] == "#"
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
