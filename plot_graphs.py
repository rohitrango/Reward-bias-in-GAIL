import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
import torch
import os
import sys
from baselines.common import plot_util as pu
from baselines.common.plot_util import COLORS

import matplotlib as mplot
mplot.rcParams.update({'font.size': 16})

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', required=True,
            help='Get filenames')
parser.add_argument('--legend', nargs='+', default=[],
            help='Legend values')
parser.add_argument('--success', default=0, type=int)
parser.add_argument('--smooth_step', default=1.0, type=float)
parser.add_argument('--length', default=0, type=int)
parser.add_argument('--max_step', default=0, type=int)
parser.add_argument('--print', default=0, type=int)
args = parser.parse_args()

def check_last_name(result):
    path = result.dirname
    splits = path.split('/')
    for sp in splits[::-1]:
        if sp == '' or sp == 'monitor':
            continue
        try:
            sp = int(sp)
        except:
            return sp
    return ''


def main():
    """
    Plot the plots inside the folder given
    """
    # Now plot the common things
    args.files = sorted(args.files)
    splits = args.files[0].split('/')
    if splits[-1] == '':
        splits = splits[-3]
    else:
        splits = splits[-2]
    env = splits
    results = []
    for file in args.files:
        print(file)
        results.extend(pu.load_results(file, success=args.success, length=args.length))

    # Print details
    if args.print:
        allrecords = dict()
        for i in range(len(results)):
            key = check_last_name(results[i])
            data = np.array(results[i].monitor)[-10:, 1]
            allrecords[key] = allrecords.get(key, []) + [data]
        # Print results
        for k, v in allrecords.items():
            v = np.concatenate(v)
            vm = v.mean()
            vs = v.std()
            print('{} {} {}'.format(k, vm, vs))
        return None

    fig = pu.plot_results(results, average_group=True,
            shaded_err=False,
            shaded_std=True,
            max_step=args.max_step,
            smooth_step=args.smooth_step,
            group_fn=lambda _: check_last_name(_),
            split_fn=lambda _: '', figsize=(10, 10))

    # Add results for behaviour cloning if present
    '''
    allfiles = []
    for file in args.files:
        for r, dirs, files in os.walk(file):
            txtfiles = list(filter(lambda x: x.endswith('bc.txt'), files))
            allfiles.extend(list(map(lambda x: os.path.join(r, x), txtfiles)))

    if allfiles != []:
        bcreward = []
        for file in allfiles:
            with open(file, 'r') as fi:
                meanrew = float(fi.readlines()[0])
                bcreward.append(meanrew)

        # Get mean and std
        mean = np.mean(bcreward)
        std = np.std(bcreward)
        idxcolor=4
        plt.plot([0, args.max_step], [mean, mean], label='BC', color=COLORS[idxcolor])
        plt.fill_between([0, args.max_step], [mean - std, mean - std], [mean + std, mean + std], alpha=0.2, color=COLORS[idxcolor])
    '''

    plt.xlabel('Number of steps', fontsize=20)
    plt.ylabel('Reward' if not args.length else 'Episode length', fontsize=20)
    #plt.yscale('log')
    plt.title(env, fontsize=24)
    if args.legend != []:
        '''
        if allfiles != []:
            args.legend.append('BC')
        '''
        #plt.legend(args.legend, loc='lower right')
        plt.legend(args.legend)
    #plt.ticklabel_format(useOffset=1)
    plt.savefig('{}.png'.format(env), bbox_inches='tight', )
    print("saved ", env)


if __name__ == "__main__":
    main()
