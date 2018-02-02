"""
[INSERT ASCII ART HERE] v0.1
Neural Unification
"""

import tensorflow as tf

# slope 10, offset 2 seems reasonable for N(0, 0.1) initalization
# fixme: the higher the dimensionality, the lower the chances of a match,
# hence slope and offset should be a function of the input dimension
# or even a trainable variable
def representation_match(rhs, goals, sigmoid_slope=10, sigmoid_offset=2):
# def representation_match(rhs, goals, sigmoid_slope=2, sigmoid_offset=0):
    """
    :param rhs: [N x k] rhs representations
    :param goals: [M x k] goal representations
    :return: [N x M] of match scores  -- previously: [M x N]???
    """
    # return tf.sigmoid(
    #     tf.matmul(rhs, goals, transpose_b=True) * sigmoid_slope -
    #     sigmoid_offset
    # )

    #rhs = tf.Print(rhs, [tf.shape(rhs)], "rhs")
    #goals = tf.Print(goals, [tf.shape(goals)], "goals")


    # dot = tf.einsum("ik,jk->ij", rhs, goals)
    # dot = (dot + 1.0) / 2.0

    def sigm_l1_sim(a, b):
        N = int(a.get_shape()[0])  # -- N x k
        M = int(b.get_shape()[0])  # -- M x k

        #a = tf.Print(a, [a], "a:\n", summarize=1000)
        #b = tf.Print(b, [b], "b:\n", summarize=1000)

        A = tf.tile(tf.expand_dims(a, 1), [1, M, 1])  # -- N x M x k
        B = tf.tile(tf.expand_dims(b, 0), [N, 1, 1])  # -- N x M x k

        #A = tf.Print(A, [A], "A:\n", summarize=1000)
        #B = tf.Print(B, [B], "B:\n", summarize=1000)

        l1 = tf.reduce_sum(tf.abs(A - B), 2)
        sim = tf.sigmoid(-l1) * 2.0

        #sim = tf.Print(sim, [sim], "sim:\n", summarize=1000)

        return sim

    def l2_sim(a, b, slope=1.0, imaginary=False):
        # using trick from: https://github.com/clinicalml/cfrnet/blob/master/cfr_net.py#L240

        #if imaginary:
        #    a, _ = tf.split(a, 2, axis=1)
        #    b, _ = tf.split(b, 2, axis=1)

        c = -2 * tf.matmul(a, tf.transpose(b))
        na = tf.reduce_sum(tf.square(a), 1, keep_dims=True)
        nb = tf.reduce_sum(tf.square(b), 1, keep_dims=True)
        # this is broadcasting!
        l2 = (c + tf.transpose(nb)) + na
        l2 = tf.clip_by_value(l2, 1e-6, 1000)
        l2 = tf.sqrt(l2)
        if slope != 1.0:
            sim = tf.exp(-l2 * slope)
        else:
            sim = tf.exp(-l2)

        # if slope != 1.0:
        #     sim = 1 / (1 + l2)
        # else:
        #     sim = 1 / (1 + slope * l2)

        #sim = tf.Print(sim, [sim], "sim:\n", summarize=1000)

        return sim

    # dot = sigm_l1_sim(rhs, goals)
    dot = l2_sim(rhs, goals)

    # dot = tf.nn.elu(dot)

    # dot = tf.nn.relu(dot)

    # dot = tf.Print(dot, [dot], "dot")

    return dot


def l2_unification(rhs, goal):
    """
    :param rhs: [N x k] rhs representations
    :param goals: [M x k] goal representations
    :return: [M x N] of match scores
    """

    pass