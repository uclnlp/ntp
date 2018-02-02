# @rockt, @timdettmers, @pminervini

import random
import re
import os
from os.path import join
import numpy as np
import copy


def clean(x):
    return x.replace(" ", "_").replace("(", "").replace(")", "").lower()

seed = 1337
rdm = np.random.RandomState(seed)
base_path = os.path.join('./data/countries')
delimiter = '\t'

acronym2name = {}
table = []

countries = set()
regions = set()
subregions = set()

country2neighbors = {}

subregion2region = {}

with open(join(base_path, "countries.csv"), "r") as f_in:
    for line in f_in.readlines()[1:]:
        line = line.strip().split(";")
        country = clean(line[0][1:-1].split(",")[0])
        acronym = line[4][1:-1]
        acronym2name[acronym] = country
        capital = line[8]
        region = clean(line[10][1:-1])
        subregion = clean(line[11][1:-1])
        borders = line[17][1:-1].split(",")
        if borders == ['']:
            borders = []

        assert country != ""

        if region != "":
            regions.add(region)
        if subregion != "":
            subregions.add(subregion)

        if region != "" and subregion != "":
            table.append((country, region, subregion, borders))
            countries.add(country)
            subregion2region[subregion] = region
    f_in.close()

facts = set()

country2region = {}
country2subregion = {}

for country, region, subregion, borders in table:
    neighbors = ["neighborOf(%s,%s).\n" % (country, acronym2name[x])
                 for x in borders]
    country2neighbors[country] = list(set(acronym2name[x] for x in borders))
    country2region[country] = region
    country2subregion[country] = subregion

assert len(countries) == 244
assert len(regions) == 5
assert len(subregions) == 23


train_dict = {}
test_candidates = []
dev_test_dict = {}

countries = list(countries)

for c in countries:
    train_dict[(c, 'subregion')] = [country2subregion[c]]
    train_dict[(c, 'region')] = [country2region[c]]
    train_dict[(c, 'neighbor')] = []
    for neighbor in country2neighbors[c]:
        train_dict[(c, 'neighbor')].append(neighbor)

    # more than one neighbor -> can be in dev or test set
    if len(country2neighbors[c]) > 0:
        test_candidates.append(c)

n = len(countries)
dev_test_size = int(n*0.1)

# splits 0.8 / 0.1 / 0.1
i = 0
violations = []
while True:
    dev_test = set(rdm.choice(test_candidates, size=dev_test_size*2, replace=False))
    contains_all_neighbor = False
    for c in dev_test:
        contains_all_neighbors = all([n in dev_test for n in country2neighbors[c]])
        print([n in dev_test for n in country2neighbors[c]])

    if not contains_all_neighbors:
        dev_test = list(dev_test)
        break
    else:
        i += 1
print(i)

dev = dev_test[:dev_test_size]
test = dev_test[dev_test_size:]

# save region data for test / dev set
dev_dict = {}
test_dict = {}
for c in dev:
    dev_dict[(c, 'region')] = train_dict[(c, 'region')]
for c in test:
    test_dict[(c, 'region')] = train_dict[(c, 'region')]


def merge_predicates(pred):
    if pred == 'region' or pred == 'subregion':
        return 'locatedIn'
    else:
        return 'neighborOf'

with open(join(base_path, "countries.nl"), "w") as f:
    for corpus, name in [(train_dict, "train")]:
                         # (dev_dict, "dev"),
                         # (test_dict, "test")]:
        #f.write("% " + "### %s ###\n" % name)
        for (country, relation), values in corpus.items():
            for value in values:
                f.write(("%s(%s,%s).\n" %
                         (merge_predicates(relation), country, value)))
    # f.write("% " + "### subregions ###\n")
    for subregion in subregion2region:
        f.write(("locatedIn(%s,%s).\n" %
                 (subregion, subregion2region[subregion])))

S1 = copy.deepcopy(train_dict)
S2 = copy.deepcopy(train_dict)
S3 = copy.deepcopy(train_dict)
for s in [dev, test]:
    for c in s:
        # remove all regions in S1
        S1.pop((c, 'region'), None)

        # remove all regions and subregion in S2
        S2.pop((c, 'region'), None)
        S2.pop((c, 'subregion'), None)

        # remove all region and subregion in S3, and additionally all regions of all neighbors of the tested country
        S3.pop((c, 'region'), None)
        S3.pop((c, 'subregion'), None)
        for n in country2neighbors[c]:
            S3.pop((n, 'region'), None)

print(len(S1.keys()))
print(len(S2.keys()))
print(len(S3.keys()))

for train, name in zip([S1, S2, S3], ['S1', 'S2', 'S3']):
    with open(join(base_path, 'countries_{0}.nl'.format(name)), 'w') as f:
        # for (country, relation), values in train.items():
        #     for value in values:
        #         # f.write(delimiter.join([country, relation, value]) + '\n')
        #         relation = merge_predicates(relation)
        #         f.write("%s(%s,%s).\n" % (relation, country, value))
        # for subregion in subregion2region:
        #     f.write(("locatedIn(%s,%s).\n" %
        #              (subregion, subregion2region[subregion])))
        # # fixme: add dev / test data!
        for corpus, name in [(train, "train")]:
            # f.write("% " + "### %s ###\n" % name)
            for (country, relation), values in corpus.items():
                for value in values:
                    f.write(("%s(%s,%s).\n" %
                             (merge_predicates(relation), country, value)))
        #f.write("% " + "### subregions ###\n")
        for subregion in subregion2region:
            f.write(("locatedIn(%s,%s).\n" %
                     (subregion, subregion2region[subregion])))


for dev_test, name in zip([dev_dict, test_dict], ['valid', 'test']):
    with open(join(base_path, '{0}.txt'.format(name)), 'w') as f:
        for (country, relation), values in dev_test.items():
            for value in values:
                # f.write(delimiter.join([country, relation, value]) + '\n')
                # relation = merge_predicates(relation)
                # f.write("%s(%s,%s).\n" % (relation, country, value))
                f.write(country + "\n")
