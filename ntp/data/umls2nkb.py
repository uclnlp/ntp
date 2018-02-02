import re
import random

atoms = []
entities = set()
with open("./data/umls/uml.db", "r") as f_in:
    for line in f_in.readlines():
        line = line.strip()
        atom = re.split("\(|\)|,", line)[1:-1]
        print(atom)
        atom = [x.lower() for x in atom if x != ""]

        entities.add(atom[1])
        entities.add(atom[2])

        if len(atom) != 0:
            atoms.append(atom)

with open("./data/umls/umls.nl", "w") as f_out:
    for atom in atoms:
        f_out.write(atom[0] + "(%s)" % ",".join(atom[1:]) + ".\n")


random.shuffle(atoms)
split = int(len(atoms) * 0.1)

train = atoms[:split * 8]
dev = atoms[split * 8:split * 9]
test = atoms[split*9:]

for name, corpus in [("train", train), ("dev", dev), ("test", test)]:
    with open("./data/umls/%s.nl" % name, "w") as f_out:
        for atom in corpus:
            f_out.write(atom[0] + "(%s)" % ",".join(atom[1:]) + ".\n")
    f_out.close()

with open("./data/umls/entities.txt", "w") as f:
    for entity in entities:
        f.write(entity + "\n")
