# coding=utf-8
"""
   _  ___________
  / |/ /_  __/ _ \
 /    / / / / ___/
/_/|_/ /_/ /_/ v0.6

Neural Theorem Provers based on Differentiable Backward Chaining

Now batched, efficient and with swag
"""

import tensorflow as tf
from ntp.util import *
import copy
import collections
from pprint import pprint
import os
from termcolor import colored, cprint
from ntp.nunify import representation_match
from ntp.jtr.util.util import nprint, tfprint
from ntp.kmax import tf_k_max

FAILURE = "FAILURE"
SUCCESS = "SUCCESS"
QUERY_VARS = "QUERY_VARS"   # track binding of vars to query representations
GOAL_DIM = "GOAL_DIM"       # track goal dimension


def is_variables_list(xs):
    if isinstance(xs, list):
        return all([is_variable(x) for x in xs])
    else:
        return False


def is_tensor(arg):
    return isinstance(arg, tf.Tensor)


def rep2string(tensor, color="magenta", show_second_dim=False):
    if show_second_dim:
        return colored("T", color) + colored("x", "blue").join([
            colored(str(x), "yellow") for x in tensor.get_shape()
        ])
    else:
        return colored("T", color) + \
               colored(str(tensor.get_shape()[0]), "yellow")


def atom2string(atom):
    atom_mapped = []
    for i, x in enumerate(atom):
        color = "magenta" if i == 0 else "cyan"
        if is_tensor(x):
            atom_mapped.append(rep2string(x, color))
        elif isinstance(x, list):
            atom_mapped.append(colored(str(x[0][0]) + str(len(x)), "white"))
        else:
            atom_mapped.append(x)
    return "%s(%s)" % (str(atom_mapped[0]),
                       colored(",", "red").join([str(x)
                                                 for x in atom_mapped[1:]]))


def rule2string(rule):
    head = atom2string(rule[0])
    body = [atom2string(x) for x in rule[1:]]
    if len(rule) == 1:
        return "%s" % head + colored(".", "red")
    else:
        return "%s" % head + colored(" :- ", "red") + \
               "%s" % ", ".join(body) + colored(".", "red")


def subs2string(substitutions):
    # return str(substitutions)
    substitutions_str = []
    for key, val in substitutions.items():
        if key == SUCCESS:
            substitutions_str = [key + ":" + rep2string(val, "cyan", True)] + \
                                substitutions_str
        elif key == GOAL_DIM or key == QUERY_VARS:
            pass
        else:
            if is_tensor(val):
                val = rep2string(val, "cyan")
            elif isinstance(val, list):
                val = colored(val[0][0] + str(len(val)), "white")
            if isinstance(key, tuple):
                key = colored(str(key[0][0]) + str(len(key)), "white")

            substitutions_str.append(str(key) + colored("/", "red") +
                                     str(val))

    return "{%s}" % ", ".join(substitutions_str)


def get_dim1(arg, substitutions=None):
    if is_tensor(arg):
        return int(arg.get_shape()[0])
    elif isinstance(arg, list):
        return len(arg)
    elif isinstance(arg, str):
        if substitutions is not None:
            if arg in substitutions:
                if not isinstance(substitutions[arg], str):
                    return int(substitutions[arg].get_shape()[0])
                else:
                    return -1
        return -1
    else:
        raise TypeError("Can't determine dim1 of %s" % str(arg))


def check_atom_consistency(atom):
    dims = [get_dim1(x) for x in atom]
    max_dim = max(dims)
    assert all([x == max_dim for x in dims]), dims


def detect_cycle(variable, substitutions):
    # cycle detection
    # todo: double-check with
    # https://github.com/gnufs/aima-java/blob/master/aima-core/src/main/java/aima/core/logic/fol/Unifier.java
    # and http://norvig.com/unify-bug.pdf
    if not isinstance(variable, list) and variable in substitutions:
        return True
    elif tuple(variable) in substitutions:
        return True
    else:
        has_cycle = False
        for key in substitutions:
            if isinstance(key, list) and variable in key:
                has_cycle = True
        return has_cycle


def unify_variable(variable, x, substitutions, depth=0):
    if detect_cycle(variable, substitutions):
        return FAILURE
    else:
        substitutions[variable] = x
        return substitutions


def unify_variables(variables, x, substitutions, depth=0):
    if detect_cycle(variables, substitutions):
        return FAILURE
    else:
        if isinstance(variables, list) and isinstance(x, list) and \
                        len(variables) == len(x):
            for var, t in zip(variables, x):
                substitutions[var] = t
        else:
            substitutions[variables] = x
        return substitutions


def batch_unify(rhs, goals, substitutions, depth=0, mask=None, transpose=False,
                nunify=representation_match, inner_tiling=True):
    current_success = nunify(rhs, goals)
    success_dim1 = int(current_success.get_shape()[1])

    if mask is not None:
        mask_dim1 = int(mask.get_shape()[1])
        if success_dim1 != mask_dim1:
            assert success_dim1 % mask_dim1 == 0
            num_mask_tiles = int(success_dim1 // mask_dim1)
            # print("inner tiling of mask", mask.get_shape(), num_mask_tiles)
            mask = inner_tile(mask, num_mask_tiles, axis=0)
        current_success = current_success * mask

    if SUCCESS in substitutions:
        old_success = substitutions[SUCCESS]
        current_success_shape = current_success.get_shape()
        old_success_shape = old_success.get_shape()

        if old_success_shape != current_success_shape:
            # old_success:              [num_rhs x num_goals]
            # old_success transpose:    [num_goals x num_rhs]
            # old_success reshape:      [1 x num_goals * num_rhs]
            old_success = tf.reshape(tf.transpose(old_success), [1, -1])
            old_success_shape = old_success.get_shape()
            assert old_success_shape[1] == current_success_shape[1], \
                str(old_success_shape) + "\t" + str(current_success_shape)
            num_tiles = int(current_success_shape[0])
            old_success = tf.tile(old_success, [num_tiles, 1])

        current_success = tf.minimum(old_success, current_success)
    substitutions[SUCCESS] = current_success
    return substitutions


def unify(rhs, goals, substitutions, depth=0, mask_id=None, transpose=False,
          nunify=representation_match, inner_tiling=True):
    substitutions_copy = copy.copy(substitutions)
    if substitutions_copy == FAILURE:
        return substitutions_copy
    elif rhs == goals:
        return substitutions_copy
    elif is_variable(rhs):
        return unify_variable(rhs, goals, substitutions_copy, depth)
    elif is_variable(goals):
        return unify_variable(goals, rhs, substitutions_copy, depth)
    elif is_variables_list(rhs):
        return unify_variables(rhs, goals, substitutions_copy, depth)
    elif is_variables_list(goals):
        return unify_variables(goals, rhs, substitutions_copy, depth)
    elif isinstance(rhs, list) and isinstance(goals, list) \
            and len(rhs) == len(goals):
        return unify(rhs[0], goals[0],
                     unify(rhs[1:], goals[1:], substitutions_copy, depth,
                           mask_id, transpose, nunify, inner_tiling),
                     depth, mask_id, transpose, nunify, inner_tiling)
    elif is_tensor(rhs) and is_tensor(goals):
        return batch_unify(rhs, goals, substitutions, depth, mask_id, transpose,
                           nunify, inner_tiling)
    else:
        return FAILURE


def substitute(goal, prev_head, substitutions, depth=0, inner_tiling=False):
    new_goal = []
    num_proofs = 0
    for j, arg in enumerate(goal):
        if is_variable(arg):
            num_proofs = max(num_proofs, get_dim1(arg, substitutions))
            if arg in substitutions:
                arg = substitutions[arg]
        new_goal.append(arg)
        num_proofs = max(num_proofs, get_dim1(goal[j], substitutions))

    new_goal = multiplex_goal(new_goal, num_proofs, inner_tiling=inner_tiling)

    return new_goal, num_proofs


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


def neural_link_predict(goals, model="ComplEx"):
    """
    :param goals: predicate, subject, object triple, each [num_goals x k]
    :param model: DistMult | ComplEx
    :return: [num_goals] scores
    """
    r, s, o = goals

    if model == "DistMult":
        raw_score = tf.einsum("ij,ij->i", r, s * o)
    elif model == "ComplEx":
        r_r, r_i = tf.split(r, 2, axis=1)
        s_r, s_i = tf.split(s, 2, axis=1)
        o_r, o_i = tf.split(o, 2, axis=1)

        score1 = tf.einsum("ij,ij->i", r_r * s_r, o_r)
        score2 = tf.einsum("ij,ij->i", r_r * s_i, o_i)
        score3 = tf.einsum("ij,ij->i", r_i * s_r, o_i)
        score4 = tf.einsum("ij,ij->i", r_i * s_i, o_r)

        raw_score = score1 + score2 + score3 - score4
    elif model == "HolE":
        s_r, s_i = tf.split(s, 2, axis=1)
        o_r, o_i = tf.split(o, 2, axis=1)

        a = tf.complex(s_r, s_i)
        b = tf.complex(o_r, o_i)

        fft_a = tf.fft(a)
        fft_b = tf.fft(b)
        fft_ac = tf.conj(fft_a)
        fft_acb = fft_ac * fft_b

        ab = tf.ifft(fft_acb)
        ab_flat = tf.concat([tf.real(ab), tf.imag(ab)], 1)

        raw_score = tf.einsum("ij,ij->i", r, ab_flat)
    else:
        raise TypeError("I don't know a neural link prediction method called ",
                        model)

    score = tf.expand_dims(tf.sigmoid(raw_score), 0)

    return score


def outer_tile(tensor, times, axis=1):
    multiples = [times, 1] if axis == 1 else [1, times]
    return tf.tile(tensor, multiples)


def inner_tile(tensor, times, axis=1):
    tensor_dims = tensor.get_shape()
    if axis == 1:
        multiples = [1, times]
        target_dim = int(tensor_dims[0]) * times
        target_shape = [target_dim, -1]
        return tf.reshape(tf.tile(tensor, multiples), target_shape)
    else:
        return tf.transpose(inner_tile(tf.transpose(tensor), times, axis=1))


def split_merge(tensor, splits, split_axis=1, merge_axis=0):
    return tf.concat(tf.split(tensor, splits, axis=split_axis), axis=merge_axis)


def tile_representations(substitutions, inner_tiling=False, in_body=False):
    success = substitutions[SUCCESS]
    success_shape = [int(x) for x in success.get_shape()]
    num_proofs = success_shape[0] * success_shape[1]
    # print("tiling subs", subs2string(substitutions), inner_tiling)
    for key in substitutions:
        if key != QUERY_VARS and key != GOAL_DIM and key != SUCCESS:
            var = substitutions[key]

            if not isinstance(var, str):
                var_dim = int(var.get_shape()[0])
                assert num_proofs % var_dim == 0
                num_tiles = num_proofs // var_dim

                if num_tiles > 1:
                    if var_dim != success_shape[1]:
                        # just got substituted in last unification,
                        # so needs to be tiled in an outer way
                        print("rep outer tiling of", key, num_tiles)
                        substitutions[key] = outer_tile(var, num_tiles)
                    else:
                        print("rep inner tiling of", key, num_tiles)
                        substitutions[key] = inner_tile(var, num_tiles)

    return substitutions


def multiplex_goal(goal, target_dim, inner_tiling=False):
    new_goal = []
    for arg in goal:
        if is_tensor(arg):
            dim1 = int(arg.get_shape()[0])
            if dim1 != target_dim:
                num_tiles = target_dim // dim1
                if inner_tiling:
                    print("goal inner tiling of", rep2string(arg), num_tiles)
                    arg = inner_tile(arg, num_tiles)
                else:
                    print("goal outer tiling of", rep2string(arg), num_tiles)
                    arg = outer_tile(arg, num_tiles)
            new_goal.append(arg)
        else:
            new_goal.append(arg)
    return new_goal


def applied_before(rule, substitutions):
    head = rule[0]
    head_vars = [x for x in head if is_variable(x)]
    return any([x for x in head_vars if x in substitutions])


def or_(nkb, goals, substitutions=dict(), depth=0, mask=None, trace=False,
        nunify=representation_match, train_0ntp=False, inner_tiling=True,
        k_max=None, max_depth=1):
    """
    :param nkb: A list of rules, which is itself a list of atoms.
    :param goals: An atom to prove.
    :param substitutions: The upstream substitutions, initially empty.
    :param depth: Depth of the prover.
    :return: List of downstream substitutions.
    """
    proofs = []

    if trace:
        print(" " * (4 * depth) + "Goal: " + atom2string(goals),
              subs2string(substitutions))

    for struct in nkb:
        rule = nkb[struct]
        head = rule[0]
        body = rule[1:]
        mask_id = None
        if mask is not None:
            mask_key, mask_id = mask
            mask_id = mask_id if mask_key == struct else None

        is_fact = len(struct) == 1 and all([not is_variable(x)
                                            for x in struct[0]])

        if not is_fact and depth == max_depth:
            # maximum depth reached
            pass
        elif not train_0ntp and is_fact and depth == 0:
            # using neural link predictor instead!
            pass
        elif applied_before(rule, substitutions):
            # rule has been applied before
            pass
        elif len(head) != len(goals):
            # unifying mismatching atoms (e.g. binary with unary predicate)
            pass
        else:
            if trace:
                print(" " * (4 * depth + 4) + "Rule: " + rule2string(rule))

            substitutions_ = unify(head, goals, substitutions, depth, mask_id,
                                   transpose=is_fact, nunify=nunify,
                                   inner_tiling=inner_tiling)

            if is_fact and k_max is not None:
                variables = [x for x in goals if is_variable(x)]
                if len(variables) > 0:
                    print(" " * (4 * depth + 8) + "Taking", k_max, "max")
                    current_success = substitutions_[SUCCESS]
                    success_k, ix_k = tf_k_max(current_success, k_max)
                    substitutions_[SUCCESS] = success_k
                    for var in variables:
                        var_rep = substitutions_[var]
                        var_rep_dim0 = int(success_k.get_shape()[1]) * k_max
                        var_rep_dim1 = int(var_rep.get_shape()[1])
                        ix_k = tf.transpose(ix_k)
                        ix_k_flat = tf.reshape(ix_k, [-1])
                        var_rep = tf.gather(var_rep, ix_k_flat,
                                            validate_indices=True)
                        var_rep.set_shape([var_rep_dim0, var_rep_dim1])
                        substitutions_[var] = var_rep

            if depth == 0 and QUERY_VARS not in substitutions_:
                query_reps = set()
                for key in substitutions_:
                    # todo: clean up
                    if is_variable(key) and not key == SUCCESS and \
                            not key == GOAL_DIM:
                        query_reps.add(key)
                substitutions_[QUERY_VARS] = query_reps

            if substitutions_ != FAILURE:
                proof = and_(nkb, body, substitutions_, head, depth, mask,
                             trace, nunify, k_max=k_max, max_depth=max_depth)

                if not isinstance(proof, list):
                    proof = [proof]
                else:
                    proof = flatten_proofs(proof)

                for proof_substitutions in proof:
                    if proof_substitutions != FAILURE:
                        proofs.append(proof_substitutions)
                        if trace:
                            print(" " * (4 * depth + 8) +
                                  colored(SUCCESS, "green") +
                                  " " + subs2string(proof_substitutions))
                    elif trace:
                        print(" " * (4 * depth + 8) + colored(FAILURE, "red"))

            elif trace:
                print(" " * (4 * depth + 8) + colored(FAILURE, "red"))
    return flatten_proofs(proofs)


def and_(nkb, subgoals, substitutions, prev_head, depth=0, mask=None,
         trace=False, nunify=representation_match, in_body=False, k_max=None,
         max_depth=1):
    """
    :param nkb: A list of rules, which is itself a list of atoms.
    :param subgoals: A list of atoms to prove.
    :param substitutions: The upstream substitutions.
    :param depth: Depth of the prover.
    :return: Downstream substitutions.
    """
    if len(subgoals) == 0:
        return substitutions
    # todo: introduce maximum depth parameter
    elif depth == max_depth:  # maximum depth
        return FAILURE
    else:
        head = subgoals[0]
        body = subgoals[1:]
        if trace:
            print(" " * (4 * depth + 4) + "Subgoal: " +
                  atom2string(head), subs2string(substitutions))

        substitutions = tile_representations(substitutions, in_body=in_body)

        proofs = []

        new_goal, num_proofs = substitute(head, prev_head, substitutions, depth,
                                          inner_tiling=in_body)
        new_body = [multiplex_goal(atom, num_proofs, inner_tiling=False) # inner_tiling=not in_body
                    for atom in body]

        for substitutions_ in or_(nkb, new_goal, substitutions, depth+1, mask,
                                  trace, nunify, inner_tiling=in_body,
                                  k_max=k_max, max_depth=max_depth):
            proofs.append(and_(nkb, new_body, substitutions_, head, depth, mask,
                               trace, nunify, in_body=True, k_max=k_max,
                               max_depth=max_depth))
        return proofs


def get_free_variables(goal):
    return [x for x in goal if is_variable(x) or is_variables_list(x)]


def aggregate_proofs(proofs, aggregation_fun=None, num_goals=1):
    tensors = [proof[SUCCESS] for proof in proofs]

    if len(tensors) == 0:
        print("WARNING! Nothing to prove!")
        return 0.0
    else:
        for i, tensor in enumerate(tensors):
            success_per_proof = tf.split(tensor, num_goals, axis=1)
            success_per_proof = [tf.reshape(x, [1, -1]) for x in success_per_proof]
            success_per_proof = tf.concat(success_per_proof, axis=0)
            tensors[i] = success_per_proof
        if len(tensors) > 1:
            success_per_proof = tf.concat(tensors, 1)
        else:
            success_per_proof = tensors[0]
        aggregated_success = aggregation_fun(success_per_proof)
        return aggregated_success


def prove(kb, goals, mask_structure, mask_var, trace=False,
          aggregation_fun=None, nunify=representation_match,
          k_max=None, train_0ntp=False, max_depth=1):
    substitutions = {GOAL_DIM: int(goals[0].get_shape()[0])}
    proofs = or_(kb, goals, substitutions,
                 mask=(mask_structure, mask_var), trace=trace, nunify=nunify,
                 k_max=k_max, train_0ntp=train_0ntp, max_depth=max_depth)
    return aggregate_proofs(proofs, aggregation_fun,
                            int(goals[0].get_shape()[0]))
