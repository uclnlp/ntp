import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# (Approximately) Differentiable K-Max
# forward
def diff_k_max(scores, k):
    """
    :param scores: [N x M]
    :param k: number of maximum elements to keep
    :return:
        [k x M] k max scores
        [k x M] indices of k max scores
    """
    #scores = np.transpose(scores)
    # print("np scores\n", scores)
    #ix = np.transpose(np.argpartition(-scores, k)[:, :k])
    # print("np ix\n", ix)
    #scores = np.transpose(-np.partition(-scores, k)[:, :k])
    # ix = np.fliplr(ix)

    ix = np.argsort(-scores, axis=0)[:k, :]
    # fixme: slow as we are sorting twice
    # replace this by indexing
    scores = -np.sort(-scores, axis=0)[:k, :]

    return scores.astype(np.float32), ix.astype(np.int32)


# backward
def diff_k_max_grad(op, grad1, grad2):
    """
    :param op:
        - scores: [N x M]
        - k
    :param
        - grad1: [k x M] upstream gradient
        - grad2: discarded
    :return: scores_grad, 0
    """
    scores = op.inputs[0]
    k = op.inputs[1]
    N = int(scores.get_shape()[0])
    M = int(scores.get_shape()[1])
    success_grad = grad1

    top_k_scores, top_k_ix = tf.nn.top_k(tf.transpose(scores), k)

    #top_k_ix = tfprint(top_k_ix, "backward scores\n", other_tensors=[top_k_scores])
    #top_k_ix = tfprint(top_k_ix, "backward ix\n")
    #ix = tf.reverse(ix, [1])
    #ix = tfprint(ix, "ix")


    # gradient scores

    # fixme: error here?
    row_ids = tf.constant([list(range(0, M))], dtype=tf.int32)
    row_ids = tf.transpose(tf.tile(row_ids, [k, 1]))
    row_ids = tf.reshape(row_ids, [-1, 1])

    sparse_ix = tf.reshape(top_k_ix, [-1, 1])
    sparse_ix = tf.concat([sparse_ix, row_ids], 1)
    sparse_ix = tf.cast(sparse_ix, tf.int64)
    sparse_grad_vals = tf.reshape(tf.transpose(success_grad), [-1])

    # validate_indices=False is unproblematic as we check that there are no
    # duplicate indices
    scores_grad_downstream = tf.sparse_to_dense(
        sparse_ix, [N, M], sparse_grad_vals,
        validate_indices=False)
    return scores_grad_downstream, tf.constant(0)


def my_py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'MyPyFuncGrad' + str(np.random.randint(0, int(1E+8)))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_k_max(scores, k, name=None):
    """
    :param scores: [N x M]
    :param k: int
    :param name:
    :return:
        - [k x M] k max scores
        - [k x M] indices of k max scores
    """
    # with tf.name_scope(name, tf_k_max, [scores, k]) as name:
    with ops.op_scope([scores, k], name, "tf_k_max") as name:
        z = my_py_func(diff_k_max, [scores, k],
                       [tf.float32, tf.int32],
                       name=name, grad=diff_k_max_grad)

        M = int(scores.get_shape()[1])
        z[0].set_shape([k, M])
        return z

if __name__ == '__main__':
    with tf.Session() as my_sess:
        N = 5
        M = 6
        k = 3
        input_size = 1

        scores = tf.Variable(np.random.randn(N, M),
                             dtype=tf.float32)

        reps = tf.constant(np.random.rand(N, input_size))

        tf.global_variables_initializer().run()

        top_k_scores, top_k_ix = tf_k_max(scores, k)

        print("in:")
        print("scores\n", scores.eval())
        print("out:")
        print("top k scores\n", top_k_scores.eval())
        print("top k ix\n", top_k_ix.eval())
        print()

        upstream_success_grad = tf.constant(
            np.random.randn(k, M) * 0.01,
            dtype=tf.float32)

        print("success grad upstream\n", upstream_success_grad.eval())
        gr_success = tf.gradients(top_k_scores, [scores, k],
                                  grad_ys=upstream_success_grad)
        print("success grad downstream\n", gr_success[0].eval())
        print()

        success_flat = tf.reshape(tf.transpose(top_k_scores), [1, -1])
        print("success flat\n", success_flat.eval())

        ix_k_flat = tf.reshape(tf.transpose(top_k_ix), [-1])
        print("ix k flat\n", ix_k_flat.eval())
        # fixme: need validate indices?
        var_rep = tf.gather(reps, ix_k_flat, validate_indices=True)
        print("reps\n", reps.eval())
        print("k reps\n", var_rep.eval())