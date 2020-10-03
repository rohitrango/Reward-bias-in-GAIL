import os
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--permonitor', type=int, default=100)

def main():
    args = parser.parse_args()
    path = args.dir
    allfiles = []
    for r, dirs, files in os.walk(path):
        files = list(filter(lambda x: x.endswith('csv'), files))
        files = list(map(lambda x: os.path.join(r, x), files))
        allfiles.extend(files)

    R, L = [], []
    for file in allfiles:
        with open(file, 'r') as fi:
            rows = list(csv.reader(fi))[-args.permonitor:]
            for row in rows:
                r = float(row[0])
                l = float(row[1])
                R.append(r)
                L.append(l)

    print(len(R), len(L))
    #print("Reward mean: {:0.2f}, Reward std: {:0.2f}".format(np.mean(R), np.std(R)))
    #print("Length mean: {:0.2f}, Length std: {:0.2f}".format(np.mean(L), np.std(L)))
    print("Reward: {:0.2f} \pm {:0.2f}".format(np.mean(R), np.std(R)))
    print("Length: {:0.2f} \pm {:0.2f}".format(np.mean(L), np.std(L)))




if __name__ == "__main__":
    main()
