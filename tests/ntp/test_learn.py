# -*- coding: utf-8 -*-

import pytest

from ntp.kb import Atom, load_from_file, normalize
from ntp.nkb import kb2nkb, augment_with_templates, embed_symbol, rule2struct
from ntp.prover import prove, representation_match, is_tensor, is_parameter, neural_link_predict
import tensorflow as tf

from ntp.jtr.train import train
from ntp.jtr.preprocess.batch import GeneratorWithRestart
from ntp.jtr.util.util import load_conf, tfprint
from ntp.experiments.util import kb_ids2known_facts

import numpy as np
import random
import copy
from tabulate import tabulate
from tensorflow.python import debug as tf_debug


def test_learn():
    conf = load_conf("./conf/synth/synth_three.conf")

    DEBUG = conf["meta"]["debug"]
    OUTPUT_PREDICTIONS = conf["meta"]["output_predictions"]
    CHECK_NUMERICS = conf["meta"]["check_numerics"]
    TFDBG = conf["meta"]["tfdbg"]
    TEST_GRAPH_CREATION = conf["meta"]["test_graph_creation"]
    TRAIN = conf["meta"]["train"]
    TEST_TIME_NEURAL_LINK_PREDICTION = conf["meta"]["test_time_neural_link_prediction"]
    TEST_TIME_BATCHING = conf["meta"]["test_time_batching"]
    ENSEMBLE = conf["meta"]["ensemble"]
    EXPERIMENT = conf["meta"]["experiment_prefix"]

    TEMPLATES_PATH = conf["data"]["templates"]

    INPUT_SIZE = conf["model"]["input_size"]
    UNIFICATION = conf["model"]["unification"]
    L2 = conf["model"]["l2"]
    UNIT_NORMALIZE = conf["model"]["unit_normalize"]
    K_MAX = conf["model"]["k_max"]
    NEURL_LINK_PREDICTOR = conf["model"]["neural_link_predictor"]
    TRAIN_0NTP = conf["model"]["train_0ntp"]
    KEEP_PROB = conf["model"]["keep_prob"]
    MAX_DEPTH = conf["model"]["max_depth"]

    TRAIN_NTP = TRAIN_0NTP or TEMPLATES_PATH is not None

    if NEURL_LINK_PREDICTOR is None and not TRAIN_0NTP:
        raise AttributeError("Can't train non-0NTP without link predictor")

    REPORT_INTERVAL = conf["training"]["report_interval"]
    NUM_EPOCHS = conf["training"]["num_epochs"]
    CLIP = conf["training"]["clip"]
    LEARNING_RATE = conf["training"]["learning_rate"]
    EPSILON = conf["training"]["epsilon"]
    OPTIMIZER = conf["training"]["optimizer"]
    POS_PER_BATCH = conf["training"]["pos_per_batch"]
    NEG_PER_POS = conf["training"]["neg_per_pos"]
    SAMPLING_SCHEME = conf["training"]["sampling_scheme"]
    MEAN_LOSS = conf["training"]["mean_loss"]
    INIT = conf["training"]["init"]

    NUM_CORRUPTIONS = 0
    if SAMPLING_SCHEME == "all":
        NUM_CORRUPTIONS = 4
    else:
        NUM_CORRUPTIONS = 2
    BATCH_SIZE = POS_PER_BATCH + POS_PER_BATCH * NEG_PER_POS * NUM_CORRUPTIONS
    kb = load_from_file(conf["data"]["kb"])

    print("Batch size: %d, pos: %d, neg: %d, corrupted: %d" %
          (BATCH_SIZE, POS_PER_BATCH, NEG_PER_POS, NUM_CORRUPTIONS))

    if TEMPLATES_PATH is not None:
        rule_templates = load_from_file(TEMPLATES_PATH, rule_template=True)
        kb = augment_with_templates(kb, rule_templates)

    kb = normalize(kb)

    nkb, kb_ids, vocab, emb, predicate_ids, constant_ids = \
        kb2nkb(kb, INPUT_SIZE, unit_normalize=UNIT_NORMALIZE, keep_prob=KEEP_PROB)

    known_facts = kb_ids2known_facts(kb_ids)

    goal_struct = rule2struct(normalize([[Atom('p1', ["c0", "c1"])]])[0])
    if EXPERIMENT == "animals":
        goal_struct = rule2struct(normalize([[Atom('p1', ["c0"])]])[0])

    def embed(goal, emb, keep_prob=1.0):
        return [embed_symbol(x, emb, unit_normalize=UNIT_NORMALIZE,
                             keep_prob=keep_prob) for x in goal]

    def get_mask_id(kb, goal_struct, goal):
        if goal_struct in kb:
            facts = kb[goal_struct][0]
            num_facts = len(facts[0])

            mask_id = None
            for i in range(num_facts):
                exists = True
                for j in range(len(goal)):
                    exists = exists and goal[j] == facts[j][i]
                if exists:
                    mask_id = i

            if mask_id is not None:
                return mask_id
        return None

    mask_indices = tf.placeholder("int32", [POS_PER_BATCH, 2], name="mask_indices")

    goal_placeholder = [tf.placeholder("int32", [BATCH_SIZE], name="goal_%d" % i)
                        for i in range(0, len(goal_struct[0]))]

    goal_emb = embed(goal_placeholder, emb, KEEP_PROB)

    num_facts = len(kb_ids[goal_struct][0][0])

    mask = tf.Variable(np.ones([num_facts, BATCH_SIZE], np.float32),
                       trainable=False, name="fact_mask")

    mask_set = tf.scatter_nd_update(mask, mask_indices, [0.0]*POS_PER_BATCH)

    mask_unset = tf.scatter_nd_update(mask, mask_indices, [1.0]*POS_PER_BATCH)

    target = tf.placeholder("float32", [BATCH_SIZE], name="target")

    AGGREGATION_METHOD = conf["model"]["aggregate_fun"]

    print(AGGREGATION_METHOD)

    if AGGREGATION_METHOD == "Max":
        def fun(x):
            return tf.reduce_max(x, 1)
        aggregation_fun = fun
    else:
        raise AttributeError("Aggregation function %s unknown" %
                             AGGREGATION_METHOD)

    def corrupt_goal(goal, args=[0], tries=100):
        if tries == 0:
            print("WARNING: Could not corrupt", goal)
            return goal
        else:
            goal_corrupted = copy.deepcopy(goal)
            for arg in args:
                corrupt = constant_ids[random.randint(0, len(constant_ids) - 1)]
                goal_corrupted[arg + 1] = corrupt

            if tuple(goal_corrupted) in known_facts:
                return corrupt_goal(goal, args, tries-1)
            else:
                return goal_corrupted

    def get_batches():
        facts = kb_ids[goal_struct][0]
        num_facts = len(facts[0])
        fact_ids = list(range(0, num_facts))

        assert num_facts >= POS_PER_BATCH

        def generator():
            random.shuffle(fact_ids)
            feed_dicts = []

            mask_indices_init = np.zeros([POS_PER_BATCH, 2], dtype=np.int32)
            goals_in_batch = [[] for _ in goal_placeholder]
            targets_in_batch = []

            j = 0
            jj = 0
            for i, ix in enumerate(fact_ids):
                current_goal = [x[ix] for x in facts]
                for k in range(len(current_goal)):
                    goals_in_batch[k].append(current_goal[k])

                targets_in_batch += [1] + [0] * (NEG_PER_POS * NUM_CORRUPTIONS)
                mask_indices_init[j] = [ix, jj]
                j += 1
                jj += 1 + (NEG_PER_POS * NUM_CORRUPTIONS)

                for _ in range(NEG_PER_POS):
                    currupt_goal_1 = corrupt_goal(current_goal, [0])
                    for k in range(len(currupt_goal_1)):
                        goals_in_batch[k].append(currupt_goal_1[k])
                    currupt_goal_2 = corrupt_goal(current_goal, [1])
                    for k in range(len(currupt_goal_2)):
                        goals_in_batch[k].append(currupt_goal_2[k])
                    if SAMPLING_SCHEME == "all":
                        currupt_goal_3 = corrupt_goal(current_goal, [0, 1])
                        for k in range(len(currupt_goal_3)):
                            goals_in_batch[k].append(currupt_goal_3[k])
                        currupt_goal_4 = corrupt_goal(current_goal, [0, 1])
                        for k in range(len(currupt_goal_4)):
                            goals_in_batch[k].append(currupt_goal_4[k])

                if j % POS_PER_BATCH == 0:
                    feed_dict = {
                        mask_indices: mask_indices_init,
                        target: targets_in_batch,
                    }
                    for k in range(len(goal_placeholder)):
                        feed_dict[goal_placeholder[k]] = goals_in_batch[k]
                    feed_dicts.append(feed_dict)
                    mask_indices_init = np.zeros([POS_PER_BATCH, 2], dtype=np.int32)
                    goals_in_batch = [[] for _ in goal_placeholder]
                    targets_in_batch = []
                    j = 0
                    jj = 0

            for f in feed_dicts:
                yield f

        return GeneratorWithRestart(generator)

    train_feed_dicts = get_batches()

    prove_success = prove(nkb, goal_emb, goal_struct, mask, trace=True,
                          aggregation_fun=aggregation_fun, k_max=K_MAX,
                          train_0ntp=TRAIN_0NTP, max_depth=MAX_DEPTH)

    print("Graph creation complete.")
    print("Variables")
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print("\t", v)

    if TEST_GRAPH_CREATION:
        exit(1)
    if DEBUG and TRAIN_NTP:
        prove_success = tfprint(prove_success, "NTP success:\n")

    def caculate_loss(success, target):
        if AGGREGATION_METHOD == "LogSumExp":
            return -(target * 2 - 1) * prove_success
        else:
            x = success
            z = target
            return - z * tf.log(tf.clip_by_value(x, EPSILON, 1.0)) - \
                   (1 - z) * tf.log(tf.clip_by_value(1 - x, EPSILON, 1.0))

    prover_loss = caculate_loss(prove_success, target)

    if DEBUG:
        prover_loss = tfprint(prover_loss, "NTP loss:\n")

    if NEURL_LINK_PREDICTOR is not None:
        neural_link_prediction_success = \
            tf.squeeze(neural_link_predict(goal_emb, model=NEURL_LINK_PREDICTOR))

        if DEBUG:
            neural_link_prediction_success = \
                tfprint(neural_link_prediction_success, "link predict:\n")

        neural_link_prediction_loss = \
            caculate_loss(neural_link_prediction_success, target)

        if TRAIN_NTP:
            loss = neural_link_prediction_loss + prover_loss
        else:
            loss = neural_link_prediction_loss
        if TEST_TIME_NEURAL_LINK_PREDICTION:
            test_time_prediction = \
                tf.maximum(neural_link_prediction_success, prove_success)

        # fixme: refactor!
        if ENSEMBLE:
            prove_success = \
                tf.maximum(neural_link_prediction_success, prove_success)
            loss = caculate_loss(prove_success, target)
    else:
        loss = prover_loss
        test_time_prediction = prove_success

    if DEBUG:
        loss = tfprint(loss, "loss:\n")

    if MEAN_LOSS:
        loss = tf.reduce_mean(loss)
    else:
        loss = tf.reduce_sum(loss)

    def pre_run(sess, epoch, feed_dict, loss, predict):
        results = sess.run(mask_set, {mask_indices: feed_dict[mask_indices]})

    def post_run(sess, epoch, feed_dict, loss, predict):
        results = sess.run(mask_unset, {mask_indices: feed_dict[mask_indices]})

    optim = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=EPSILON)

    gradients = optim.compute_gradients(loss)
    variables = [x[1] for x in gradients]
    gradients = [x[0] for x in gradients]

    hooks = []

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    sess = tf.Session(config=session_config)

    if TFDBG:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    if TRAIN:
        train(loss, optim, train_feed_dicts, max_epochs=NUM_EPOCHS,
              hooks=hooks, pre_run=pre_run, post_run=post_run, sess=sess, l2=L2,
              clip=CLIP, check_numerics=CHECK_NUMERICS)
    else:
        sess.run(tf.global_variables_initializer())

    predicate_ids_with_placholders = copy.deepcopy(predicate_ids)
    predicate_ids = []
    for id in predicate_ids_with_placholders:
        if not is_parameter(vocab.id2sym[id]):
            predicate_ids.append(id)

    if OUTPUT_PREDICTIONS:
        goal_placeholder = [
            tf.placeholder("int32", [1], name="goal_%d" % i)
            for i in range(0, len(goal_struct[0]))]

        goal_emb = embed(goal_placeholder, emb, keep_prob=1.0)

        if TEST_TIME_BATCHING:
            copies = BATCH_SIZE
            for i, x in enumerate(goal_emb):
                goal_emb[i] = tf.tile(x, [copies, 1])

        prove_success_test_time = \
            prove(nkb, goal_emb, goal_struct, mask_var=None, trace=True,
                  aggregation_fun=aggregation_fun, k_max=K_MAX,
                  train_0ntp=TRAIN_0NTP, max_depth=MAX_DEPTH)

        if NEURL_LINK_PREDICTOR is not None:
            neural_link_prediction_success_test_time = \
                neural_link_predict(goal_emb, model=NEURL_LINK_PREDICTOR)
            if TEST_TIME_NEURAL_LINK_PREDICTION:
                prove_success_test_time = \
                    tf.maximum(prove_success_test_time,
                               neural_link_prediction_success_test_time)

        table = []
        for sym in vocab.sym2id:
            id = vocab.sym2id[sym]
            vec = sess.run(emb[id])
            table.append([sym, id, vec])

        def predict(predicate, arg1, arg2):
            feed_dict = {}

            goal = [vocab(predicate), vocab(arg1), vocab(arg2)]

            for k, d in zip(goal_placeholder, goal):
                feed_dict[k] = [d]

            success = prove_success_test_time

            if AGGREGATION_METHOD == "LogSumExp":
                success = tf.sigmoid(success)

            if TEST_TIME_NEURAL_LINK_PREDICTION:
                success = tf.squeeze(success)

            success_val = sess.run(success, feed_dict=feed_dict)

            if TEST_TIME_BATCHING:
                if not all([x == success_val[0] for x in success_val]):
                    print("WARNING! Numerical instability?", success_val)

            return success_val

        table = []
        headers = [vocab.id2sym[rid] for rid in predicate_ids]
        for i, e1id in enumerate(constant_ids):
            for j, e2id in enumerate(constant_ids):
                e1 = vocab.id2sym[e1id]
                e2 = vocab.id2sym[e2id]
                row = [e1, e2]
                for r in headers:
                    score = predict(r, e1, e2)
                    if TEST_TIME_BATCHING:
                        score = score[0]
                    row.append(score)
                table.append(row)
        print(tabulate(table, headers=["e1", "e2"] + headers))

    def decode(x, emb, vocab, valid_ids, sess):
        valid_ids = set(valid_ids)

        num_rules = int(x.get_shape()[0])
        num_symbols = int(emb.get_shape()[0])

        mask = np.ones([num_symbols], dtype=np.float32)
        for i in range(len(vocab)):
            if i not in valid_ids:
                mask[i] = 0

        # -- num_rules x num_symbols
        mask = tf.tile(tf.expand_dims(mask, 0), [num_rules, 1])

        # -- num_rules x num_symbols
        match = representation_match(x, emb)
        success_masked = match * mask

        success, ix = tf.nn.top_k(success_masked, 1)
        success_val, ix_val = sess.run([success, ix], {})

        syms = []
        for i, row in enumerate(ix_val):
            sym_id = row[0]
            sym_success = success_val[i][0]
            sym = vocab.id2sym[sym_id]
            syms.append((sym, sym_success))
        return syms

    def unstack_rules(rule):
        rules = []
        num_rules = len(rule[0].predicate)
        for i in range(num_rules):
            current_rule = []
            confidence = 1.0
            for atom in rule:
                predicate = atom.predicate
                if isinstance(predicate, list):
                    predicate, success = predicate[i]
                    confidence = min(confidence, success)
                arguments = []
                for argument in atom.arguments:
                    if isinstance(argument, list):
                        argument, success = argument[i]
                        arguments.append(argument)
                        confidence = min(confidence, success)
                    else:
                        arguments.append(argument)
                current_rule.append(Atom(predicate, arguments))
            rules.append((current_rule, confidence))
        return rules

    for struct in nkb:
        if len(struct) > 1:
            rule = nkb[struct]
            rule_sym = []
            for atom in rule:
                atom_sym = []
                for i, sym in enumerate(atom):
                    if is_tensor(sym):
                        valid_ids = predicate_ids if i == 0 else constant_ids
                        atom_sym.append(decode(sym, emb, vocab, valid_ids, sess))
                    else:
                        atom_sym.append(sym[0])
                rule_sym.append(Atom(atom_sym[0], atom_sym[1:]))

            rules = unstack_rules(rule_sym)

            rules.sort(key=lambda x: - x[1])

            max_a, max_b = 0.0, 0.0
            for rule, confidence in rules:
                head, body = rule[0], rule[1]

                if head.predicate == 'p' and body.predicate == 'q' and head.arguments == body.arguments:
                    max_a = max(max_a, confidence)
                elif head.predicate == 'q' and body.predicate == 'p' and head.arguments == body.arguments:
                    max_b = max(max_b, confidence)

            assert max_a > 0.9
            assert max_b > 0.9

if __name__ == '__main__':
    pytest.main([__file__])
