# -*- coding: utf-8 -*-

from ntp.kb import load_from_file
from ntp.nkb import kb2nkb

import pytest


def test_kb():
    kb = load_from_file("./data/synth/one.nl")

    for i, entry in enumerate(kb):
        assert len(entry) == 1
        atom = entry[0]
        assert atom.predicate == 'p'
        assert atom.arguments == ['e{}'.format(i), 'e{}'.format(i)]

    INPUT_SIZE = 100
    UNIT_NORMALIZE = False
    KEEP_PROB = 1.0

    nkb, kb_ids, vocab, emb, predicate_ids, constant_ids = \
        kb2nkb(kb, INPUT_SIZE, unit_normalize=UNIT_NORMALIZE,
               keep_prob=KEEP_PROB)

    for k, v in kb_ids.items():
        assert k == (('p0', 'c', 'c'),)
        a, b, c = v[0]
        assert a.shape == b.shape == c.shape == (33,)
        print(b, c)

if __name__ == '__main__':
    pytest.main([__file__])
