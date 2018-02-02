"""
[INSERT ASCII ART HERE] v0.1
Symbolic Representation of a Knowledge Base
"""

import re
from pprint import pprint
from ntp.util import trim, is_variable, has_free_variables, Atom


def parse_rules(rules, delimiter="#####", rule_template=False):
    """
    :param rules:
    :param delimiter:
    :return:
    """
    kb = []
    for rule in rules:
        if rule_template:
            splits = re.split("\A\n?([0-9]?[0-9]+)", rule)
            # fixme: should be 0 and 1 respectively
            num = int(splits[1])
            rule = splits[2]
        rule = re.sub(":-", delimiter, rule)
        rule = re.sub("\),", ")"+delimiter, rule)
        rule = [trim(x) for x in rule.split(delimiter)]
        rule = [x for x in rule if x != ""]
        if len(rule) > 0:
            atoms = []
            for atom in rule:
                splits = atom.split("(")
                predicate = splits[0]
                args = [x for x in re.split("\s?,\s?|\)", splits[1]) if x != ""]
                atoms.append(Atom(predicate, args))
            if rule_template:
                kb.append((atoms, num))
            else:
                kb.append(atoms)
    return kb


def load_from_file(path, rule_template=False):
    with open(path, "r") as f:
        text = f.readlines()
        text = [x for x in text if not x.startswith("%") and x.strip() != ""]
        text = "".join(text)
        rules = [x for x in re.split("\.\n|\.\Z", text) if x != "" and
                 x != "\n" and not x.startswith("%")]
        kb = parse_rules(rules, rule_template=rule_template)
        # pprint(kb)
        return kb


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


if __name__ == '__main__':
    kb = load_from_file("./data/ntp/legal.nl")
    pprint(kb)
    for rule in kb:
        for atom in rule:
            print(atom.predicate)
            print(atom.arguments)
