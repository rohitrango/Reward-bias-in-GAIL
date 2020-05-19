import argparse
import pickle as pkl
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='File path', required=True)

args = parser.parse_args()

def main():
    with open(args.file, 'rb') as fi:
        data = pkl.load(fi)
    newdata = list(filter(lambda x: x.rews.sum() > 0, data))
    print(len(data), len(newdata))
    with open(args.file, 'wb') as fi:
        pkl.dump(newdata, fi)
    print("Saved {}".format(args.file))

if __name__ == "__main__":
    main()
