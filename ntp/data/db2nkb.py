import re
import random

random.seed(1337)

# for dat in ["umls", "kinship"]:
# for dat in ["kinship", "nations"]:
# for dat in ["animals"]:
for dat in ["nations"]:
    db_name = dat[:3]
    atoms = []
    entities = set()

    with open("./data/%s/%s.db" % (dat, db_name), "r") as f_in:
        for line in f_in.readlines():
            line = line.strip()
            atom = re.split("\(|\)|,", line)[1:-1]
            print(atom)
            atom = [x.lower() for x in atom if x != ""]
            if len(atom) > 2:
                entities.add(atom[1])
                entities.add(atom[2])
            # else:
            #     atom = atom[::-1]
            #
            # if len(atom) != 0:
                atoms.append(atom)

    with open("./data/%s/%s.nl" % (dat, dat), "w") as f_out:
        for atom in atoms:
            f_out.write(atom[0] + "(%s)" % ",".join(atom[1:]) + ".\n")

    random.shuffle(atoms)
    split = int(len(atoms) * 0.1)

    train = atoms[:split * 8]
    dev = atoms[split * 8:split * 9]
    test = atoms[split*9:]

    for name, corpus in [("train", train), ("dev", dev), ("test", test)]:
        with open("./data/%s/%s.nl" % (dat, name), "w") as f_out:
            for atom in corpus:
                f_out.write(atom[0] + "(%s)" % ",".join(atom[1:]) + ".\n")
        f_out.close()

    with open("./data/%s/entities.txt" % dat, "w") as f:
        for entity in entities:
            f.write(entity + "\n")
