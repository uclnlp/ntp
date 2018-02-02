from ntp.kb import Atom, load_from_file, normalize
from ntp.nkb import kb2nkb, augment_with_templates, embed_symbol, rule2struct
from ntp.prover import prove, representation_match, is_tensor, is_parameter, neural_link_predict
from ntp.tp import rule2string
import tensorflow as tf
from pprint import pprint
from ntp.jtr.train import train
from ntp.jtr.util.hooks import LossHook, ExamplesPerSecHook, ETAHook, TensorHook
from ntp.jtr.preprocess.batch import GeneratorWithRestart
from ntp.jtr.util.util import nprint, get_timestamped_dir, load_conf, save_conf, tfprint
import numpy as np
import random
import copy
from tensorflow.python import debug as tf_debug
import sys
import itertools

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

tf.set_random_seed(1337)
np.random.seed(1337)


def kb_ids2known_facts(kb_ids):
    """
    :param kb_ids: a knowledge base of facts that are already mapped to ids
    :return: a set of all known facts (used later for negative sampling)
    """

    facts = set()
    for struct in kb_ids:
        arrays = kb_ids[struct][0]
        num_facts = len(arrays[0])
        for i in range(num_facts):
            fact = [x[i] for x in arrays]
            facts.add(tuple(fact))
    return facts


if __name__ == '__main__':
    if len(sys.argv) > 1:
        conf = load_conf(sys.argv[1])
    else:
        conf = load_conf("./conf/wordnet.conf")

    DIR = "/".join(sys.argv[1].split("/")[:-1])

    experiment_prefix = conf["meta"]["experiment_prefix"]
    experiment_dir = get_timestamped_dir("./out/" + experiment_prefix,
                                         conf["meta"]["name"],
                                         link_to_latest=True)
    save_conf(experiment_dir + conf["meta"]["file_name"], conf)

    pprint(conf)

    DEBUG = conf["meta"]["debug"]
    OUTPUT_PREDICTIONS = conf["meta"]["output_predictions"]
    CHECK_NUMERICS = conf["meta"]["check_numerics"]
    TFDBG = conf["meta"]["tfdbg"]
    TEST_GRAPH_CREATION = conf["meta"]["test_graph_creation"]
    TRAIN = False
    TEST_TIME_NEURAL_LINK_PREDICTION = \
        conf["meta"]["test_time_neural_link_prediction"]
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

    permutations = itertools.permutations([0, 1, 2, 3])

    for permutation in permutations:
        tf.reset_default_graph()
        nkb, kb_ids, vocab, emb, predicate_ids, constant_ids = \
            kb2nkb(kb, INPUT_SIZE, unit_normalize=UNIT_NORMALIZE,
                   keep_prob=KEEP_PROB, permutation=permutation)

        pprint(nkb.keys())
        pprint(vocab.sym2id)

        known_facts = kb_ids2known_facts(kb_ids)

        # using same embedding matrix
        # tf.get_variable_scope().reuse_variables()
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

        mask_set = tf.scatter_nd_update(mask, mask_indices, [0.0] * POS_PER_BATCH)

        mask_unset = tf.scatter_nd_update(mask, mask_indices, [1.0] * POS_PER_BATCH)

        target = tf.placeholder("float32", [BATCH_SIZE], name="target")

        AGGREGATION_METHOD = conf["model"]["aggregate_fun"]
        aggregation_fun = None
        if AGGREGATION_METHOD == "Max":
            def fun(x):
                return tf.reduce_max(x, 1)


            aggregation_fun = fun
        elif AGGREGATION_METHOD == "Mean":
            def fun(x):
                return tf.reduce_mean(x, 1)


            aggregation_fun = fun
        elif AGGREGATION_METHOD == "LogSumExp":
            # fixme: problem since in our case this is already applied to sub-paths
            def fun(x):
                return tf.reduce_logsumexp(x, 1)


            aggregation_fun = fun
        elif AGGREGATION_METHOD == "MaxMean":
            def fun(x):
                return (tf.reduce_max(x, 1) + tf.reduce_mean(x, 1)) / 2.0


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
                    return corrupt_goal(goal, args, tries - 1)
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

        # for _ in range(6):
        #     for x in train_feed_dicts:
        #         for key in x:
        #             val = x[key]
        #             print(key, val)
        #         print()
        #     print("---")
        # os._exit(-1)

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
                return -z * tf.log(tf.clip_by_value(x, EPSILON, 1.0)) - \
                       (1 - z) * tf.log(tf.clip_by_value(1 - x, EPSILON, 1.0))
                # using numerical stable implementation from tf.nn.sigmoid_cross_entropy_with_logits
                # loss = tf.maximum(x, 0) - x * target + tf.log(1 + tf.exp(-tf.abs(x)))


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


        # loss = tf.reduce_sum(loss)

        # loss = tfprint(loss, "loss reduced:\n")


        def pre_run(sess, epoch, feed_dict, loss, predict):
            results = sess.run(mask_set, {mask_indices: feed_dict[mask_indices]})
            if DEBUG:
                # for id in vocab.id2sym:
                #     print(id, vocab.id2sym[id])
                # print("mask\n", results)
                for k in feed_dict:
                    print(k, feed_dict[k])
                pass


        def post_run(sess, epoch, feed_dict, loss, predict):
            results = sess.run(mask_unset, {mask_indices: feed_dict[mask_indices]})
            # print(results)
            if DEBUG:
                exit(1)
                pass


        summary_writer = tf.summary.FileWriter(experiment_dir)

        optim = None
        if OPTIMIZER == "Adam":
            optim = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
                                           epsilon=EPSILON)
        if OPTIMIZER == "AdaGrad":
            optim = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
        elif OPTIMIZER == "SGD":
            optim = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

        gradients = optim.compute_gradients(loss)
        variables = [x[1] for x in gradients]
        gradients = [x[0] for x in gradients]

        hooks = [
            LossHook(REPORT_INTERVAL, 1, summary_writer=summary_writer),
            ExamplesPerSecHook(REPORT_INTERVAL, BATCH_SIZE,
                               summary_writer=summary_writer),
            ETAHook(REPORT_INTERVAL, NUM_EPOCHS, 10, summary_writer=summary_writer)
        ]

        if DEBUG:
            hooks.append(
                TensorHook(REPORT_INTERVAL, variables, prefix="variables_",
                           modes=["mean_abs", "std", "norm", "max", "min"],
                           global_statistics=True, summary_writer=summary_writer))
            hooks.append(
                TensorHook(REPORT_INTERVAL, gradients, prefix="gradients_",
                           modes=["mean_abs", "std", "norm", "max", "min"],
                           global_statistics=True, summary_writer=summary_writer))

        sess = tf.Session()
        if TFDBG:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        if TRAIN:
            train(loss, optim, train_feed_dicts, max_epochs=NUM_EPOCHS,
                  hooks=hooks, pre_run=pre_run, post_run=post_run, sess=sess, l2=L2,
                  clip=CLIP, check_numerics=CHECK_NUMERICS)
        else:
            # todo: load model
            print("Loading model...")
            # tf.saved_model.loader.load(sess, ["foo"], DIR)
            tf.train.Saver().restore(sess, DIR + "/model.ckpt-0")
            # sess.run(tf.global_variables_initializer())
            print(emb.eval(sess))


        def decode(x, emb, vocab, valid_ids, sess):
            valid_ids = set(valid_ids)

            num_rules = int(x.get_shape()[0])
            num_symbols = int(emb.get_shape()[0])

            mask = np.ones([num_symbols], dtype=np.float32)
            for i in range(len(vocab)):
                if i not in valid_ids:  # or i == vocab.sym2id[vocab.unk]:
                    mask[i] = 0  # np.zeros([input_size], dtype=np.float32)

            # -- num_rules x num_symbols
            mask = tf.tile(tf.expand_dims(mask, 0), [num_rules, 1])
            # print(sess.run(mask[0], {}))

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


        predicate_ids_with_placholders = copy.deepcopy(predicate_ids)
        predicate_ids = []
        for id in predicate_ids_with_placholders:
            if not is_parameter(vocab.id2sym[id]):
                predicate_ids.append(id)

        print("Writing induced logic program to", experiment_dir + "rules.nl")
        with open(experiment_dir + "rules_%s.nl" % ("".join([str(x) for x in permutation])), "w") as f:
            for struct in nkb:
                # it's a rule
                if len(struct) > 1:
                    rule = nkb[struct]
                    rule_sym = []
                    for atom in rule:
                        atom_sym = []
                        for i, sym in enumerate(atom):
                            if is_tensor(sym):
                                valid_ids = predicate_ids if i == 0 else constant_ids
                                syms = decode(sym, emb, vocab, valid_ids, sess)
                                print(syms)
                                atom_sym.append(syms)
                            else:
                                atom_sym.append(sym[0])
                        rule_sym.append(Atom(atom_sym[0], atom_sym[1:]))

                    rules = unstack_rules(rule_sym)

                    rules.sort(key=lambda x: -x[1])

                    # filtering for highly confident rules
                    # rules = [rule for rule in rules if rule[1] > 0.8]

                    f.write(str(struct) + "\n")
                    for rule, confidence in rules:
                        f.write("%s\t%s\n" % (confidence, rule2string(rule)))
                    f.write("\n")
            f.close()
