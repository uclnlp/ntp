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
