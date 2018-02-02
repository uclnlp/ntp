import random
from ntp.kb import Atom, load_from_file, normalize

random.seed(1337)


def sample_kb(file_path, constants, use_pairs, predicates, facts,
              implications=0,
              inversions=0,
              symmetries=0,
              transitivities=0,
              templates=0):
    constant_symbols = ["e%d" % (i+1) for i in range(constants)]
    constant_pairs = []
    if use_pairs:
        for i in range(constants):
            constant_pairs.append(
                (constant_symbols[random.randint(0, len(constant_symbols) - 1)],
                 constant_symbols[random.randint(0, len(constant_symbols) - 1)])
            )

    predicate_symbols = ["p%d" % (i+1) for i in range(predicates)]

    kb = set()

    if templates > 0:
        with open(file_path + ".nlt", "w") as f_templates:
            if implications > 0:
                f_templates.write("%d\t#1(X,Y) :- #2(X,Y).\n" % templates)
            if inversions > 0:
                f_templates.write("%d\t#1(X,Y) :- #2(Y,X).\n" % templates)
            if symmetries > 0:
                f_templates.write("%d\t#1(X,Y) :- #1(Y,X).\n" % templates)
            if transitivities > 0:
                f_templates.write("%d\t#1(X,Y) :- #2(X,Z), #2(Z,Y).\n" % templates)

    with open(file_path+".nl", "w") as f:
        for i in range(facts):
            e1 = None
            e2 = None
            if use_pairs:
                e1, e2 = constant_pairs[random.randint(0, len(constant_pairs) - 1)]
            else:
                e1 = constant_symbols[random.randint(0, len(constant_symbols) - 1)]
                e2 = constant_symbols[random.randint(0, len(constant_symbols) - 1)]

            r = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
            rule = "%s(%s,%s).\n" % (r, e1, e2)
            if rule not in kb:
                kb.add(rule)

        for i in range(implications):
            def sample_rule(tries=10):
                r = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
                s = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
                rule = "%s(X,Y) :- %s(X,Y).\n" % (r, s)
                if s != r and rule not in kb:
                    kb.add(rule)
                else:
                    sample_rule(tries-1)
            sample_rule(10)

        for i in range(inversions):
            def sample_rule(tries=10):
                r = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
                s = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
                rule = "%s(X,Y) :- %s(Y,X).\n" % (r, s)
                if s != r and rule not in kb:
                    kb.add(rule)
                else:
                    sample_rule(tries-1)
            sample_rule(10)

        for i in range(symmetries):
            def sample_rule(tries=10):
                r = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
                rule = "%s(X,Y) :- %s(Y,X).\n" % (r, r)
                if rule not in kb:
                    kb.add(rule)
                else:
                    sample_rule(tries-1)
            sample_rule(10)

        for i in range(transitivities):
            def sample_rule(tries=10):
                r = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
                s = predicate_symbols[random.randint(0, len(predicate_symbols) - 1)]
                rule = "%s(X,Y) :- %s(X,Z), %s(Z,Y).\n" % (r, s, s)
                if rule not in kb:
                    kb.add(rule)
                else:
                    sample_rule(tries-1)
            sample_rule(10)

        for rule in kb:
            f.write(rule)
        f.close()


def rule2str(rule):
    if len(rule) == 2:
        body, head = rule
        return ("%s(%s,%s) :- %s(%s,%s)." %
                      (head[0], head[1], head[2],
                       body[0], body[1], body[2]))
    else:
        body1, body2, head = rule
        return (
            "%s(%s,%s) :- %s(%s, %s), %s(%s,%s)." %
            (head[0], head[1], head[2],
             body1[0], body1[1], body1[2],
             body2[0], body2[1], body2[2]))


def infer(file_path, hops=2):
    kb = normalize(load_from_file(file_path+".nl"))

    def is_fact(x):
        return len(x) == 1

    def atom2tuple(atom):
        return atom.predicate, atom.arguments[0], atom.arguments[1]

    facts = set()
    for x in kb:
        if is_fact(x):
            facts.add(atom2tuple(x[0]))

    rules = []
    for x in kb:
        if not is_fact(x):
            if len(x) == 2:
                rules.append((atom2tuple(x[1]), atom2tuple(x[0])))
            else:
                rules.append(
                    (atom2tuple(x[1]), atom2tuple(x[2]), atom2tuple(x[0])))

    inferred_facts = facts.copy()
    z = 0
    while hops > 0:
        hops -= 1
        z += 1
        print("Iter", z)
        new_facts = inferred_facts.copy()
        for rule in rules:
            if len(rule) == 2:
                r = rule[0][0]
                s = rule[1][0]
                if r == s and rule[0][1] == rule[1][2]:
                    # symmetry
                    for (a, e1, e2) in inferred_facts:
                        if a == r and (s, e2, e1) not in inferred_facts:
                            print("Inferred %s(%s,%s) using %s" % (s, e2, e1, rule2str(rule)))
                            new_facts.add((s, e2, e1))
                elif r != s and rule[0][1] == rule[1][2]:
                    # inversion
                    for (a, e1, e2) in inferred_facts:
                        if a == r and (s, e2, e1) not in inferred_facts:
                            print("Inferred %s(%s,%s) using %s" % (s, e2, e1, rule2str(rule)))
                            new_facts.add((s, e2, e1))
                else:
                    # implication
                    for (a, e1, e2) in inferred_facts:
                        if a == r and (s, e1, e2) not in inferred_facts:
                            print("Inferred %s(%s,%s) using %s" % (s, e1, e2, rule2str(rule)))
                            new_facts.add((s, e1, e2))
            else:
                # transitivity
                r = rule[0][0]
                s = rule[1][0]
                t = rule[2][0]
                for (a, e1, e2) in inferred_facts:
                    if a == r:
                        for (b, e3, e4) in inferred_facts:
                            if b == s and e2 == e3 and (t, e1, e4) not in inferred_facts:
                                print("Inferred %s(%s,%s) using %s" % (t, e1, e4, rule2str(rule)))
                                new_facts.add((t, e1, e4))

        inferred_facts = new_facts

        if hops == 0 or len(new_facts) == len(inferred_facts):
            inferred_facts -= facts
            with open(file_path + "_facts.nl", "w") as f_facts:
                for (r, e1, e2) in facts:
                    f_facts.write("%s(%s,%s).\n" % (r, e1, e2))
                for (r, e1, e2) in inferred_facts:
                    f_facts.write("%s(%s,%s).\n" % (r, e1, e2))
                f_facts.close()

            with open(file_path+"_train.nl", "w") as f_train:
                for (r, e1, e2) in facts:
                    f_train.write("%s(%s,%s).\n" % (r, e1, e2))
                f_train.close()

            with open(file_path + "_test.nl", "w") as f_test:
                for (r, e1, e2) in inferred_facts:
                    f_test.write("%s(%s,%s).\n" % (r, e1, e2))
                f_test.close()

            with open(file_path + "_rules.nl", "w") as f_rules:
                for rule in rules:
                    f_rules.write(rule2str(rule)+"\n")
                f_rules.close()
            return


if __name__ == '__main__':
    constants = 20
    predicates = 5
    facts = 100
    hops = 1
    templates = 3

    for file_path, implications, inversions, symmetries, transitivities in \
        [
            ("./data/sampled/facts", 0, 0, 0, 0),
            ("./data/sampled/implications", 1, 0, 0, 0),
            ("./data/sampled/inversions", 0, 1, 0, 0),
            ("./data/sampled/symmetries", 0, 0, 1, 0),
            ("./data/sampled/transitivities", 0, 0, 0, 1),
            ("./data/sampled/implications_inversions", 1, 1, 0, 0),
            ("./data/sampled/all", 1, 1, 1, 1)
        ]:
        print("generating", file_path)
        sample_kb(file_path,
                  constants=20,
                  use_pairs=False,
                  predicates=4,
                  facts=50,
                  implications=implications,
                  inversions=inversions,
                  symmetries=symmetries,
                  transitivities=transitivities,
                  templates=templates)
        infer(file_path, hops=1)
        print()




