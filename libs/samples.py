#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import glob
import os
import random
from collections import defaultdict

import numpy as np

from libs.util.misc import split_prefix_number


def get_samples(input_root, ext):
    fn_pattern = os.path.join(input_root, "t*__*_*__*.{}".format(ext))
    fns = glob.glob(fn_pattern)
    names = defaultdict(dict)
    for fn in fns:
        name = os.path.basename(fn).rstrip(".{}".format(ext))
        parts = name.split('__')
        item = parts[-1]
        key = '__'.join(parts[:-1])
        names[key][item] = fn
    return names


def create_split(samples, fraction=0.8):
    split = defaultdict(list)
    tasks = defaultdict(list)
    for name, e in samples.items():
        task = name.split('__')[0]
        tasks[task].append(name)
    for task, task_names in tasks.items():
        random.shuffle(task_names)
        n_names = len(task_names)
        split_idx = int(n_names * fraction)
        if split_idx == 0:
            raise RuntimeError("Not enough elements to create a {:.0%} (train) split: {}".format(fraction, n_names))
        split['train'].extend(task_names[:split_idx])
        split['test'].extend(task_names[split_idx:])
    return split


def enumerate_names(task_names: dict, shuffle=True, group=True, rand: np.random.RandomState=None):
    rand = np.random.RandomState() if rand is None else rand
    if group and shuffle:
        tasks = list(task_names.keys())
        task_iters = dict((k, iter([])) for k, v in task_names.items())
        while True:
            rand.shuffle(tasks)
            for task in tasks:
                name = None
                retry = True
                while name is None:
                    it = task_iters[task]
                    name = next(it, None)
                    if name is None:
                        if retry:
                            v = list(task_names[task])
                            rand.shuffle(v)
                            task_iters[task] = iter(v)
                            retry = False
                        else:
                            raise RuntimeError("No names for task: {}".format(task))
                    else:
                        break
                yield name
    else:
        names = list(n for l in task_names.values() for n in l)
        if shuffle:
            rand.shuffle(names)
        yield from names


def get_sample_info(root, ext='nrrd'):
    sample_pattern = os.path.join(root, '*.{}'.format(ext))
    sample_fns = sorted(glob.glob(sample_pattern))
    sample_map = defaultdict(dict)
    sample_sets = defaultdict(set)
    sample_meta = defaultdict(dict)
    for sample_fn in sample_fns:
        sample_name = os.path.splitext(os.path.basename(sample_fn))[0]
        sample_parts = sample_name.split('__')
        sample_prefix = sample_parts[0]
        sample_key = sample_parts[-1]
        sample_set, sample_group = sample_prefix.split('-')
        sample_set_key = split_prefix_number(sample_set)[0]

        sample_id = '__'.join(sample_parts[0:-1])
        sample_map[sample_id][sample_key] = sample_fn
        sample_sets[sample_set_key].add(sample_id)
        if sample_id not in sample_meta:
            sample_meta[sample_id] = {
                'set': sample_set_key,
                'group': sample_group,
            }
    return sample_map, sample_sets, sample_meta
