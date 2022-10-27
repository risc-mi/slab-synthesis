#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import glob
import os
from collections import OrderedDict, defaultdict
from typing import Optional

import numpy as np


def get_settings(base_spacing=None):
    task_count = 10
    template = {
        'scale': 1.0,
        'background': 0,
        'masking': 'threshold',
        'threshold': 0
    }
    settings = dict((i + 1, dict(template)) for i in range(task_count))

    # BRATS
    s = settings[1]
    s['scale'] = 0.5

    # Hearth
    s = settings[2]
    s['scale'] = 1
    s['masking'] = 'contour'
    s['radius'] = 2
    s['threshold'] = 60

    # Liver
    s = settings[3]
    s['scale'] = 1
    s['masking'] = 'contour'
    s['radius'] = 0
    s['threshold'] = -500
    s['background'] = -1000

    # Hippocampus
    s = settings[4]
    s['scale'] = 0.15

    # Prostate
    s = settings[5]
    s['scale'] = 0.5
    s['masking'] = 'none'

    # Lung
    s = settings[6]
    s['scale'] = 1
    s['masking'] = 'contour'
    s['radius'] = 0
    s['threshold'] = -200
    s['background'] = -1000

    # Pancreas
    s = settings[7]
    s['scale'] = 1
    s['masking'] = 'contour'
    s['radius'] = 0
    s['threshold'] = -200
    s['background'] = -1000

    # Haptic Vessel
    s = settings[8]
    s['scale'] = 1
    s['masking'] = 'contour'
    s['radius'] = 0
    s['threshold'] = -200
    s['background'] = -1000

    # Spleen
    s = settings[9]
    s['scale'] = 1
    s['masking'] = 'contour'
    s['radius'] = 0
    s['threshold'] = -200
    s['background'] = -1000

    # Colon
    s = settings[10]
    s['scale'] = 1
    s['masking'] = 'contour'
    s['radius'] = 0
    s['threshold'] = -200
    s['background'] = -1000

    # calculate spacings
    if base_spacing is not None:
        for task, setting in settings.items():
            setting['spacing'] = np.multiply(base_spacing, setting['scale']).tolist()
    return settings


def _default_parse_name_tags(name: str, source: str):
    """
    default method to parse an item name structured in tags, the last tag indicates the source
    :param name: name to parse
    :param source: default source name
    :return: parsed name and source
    """
    parts = name.split('-')
    name_part = '-'.join(parts[:-1])
    if name_part:
        source = '{}.{}'.format(source, parts[-1])
        name = name_part
    return name, source


def enumerator(sources: dict, ext: Optional[str]=None, prefix='pat', tags=False, parse_fn=None):
    """
    enumerates equally named items in different named paths/folders
    Example:
    lets have to folders C:\step1 and C:\step2, each containing images imgA.png and imgB.png
    calling enumerator({"step1": r"C:\step1", "step2": r"C:\step2"}, 'png', prefix='img')
    will yield two times:
    - {"step1": r"C:\step1\imgA.png", "step2": r"C:\step2\imgA.png"}
    - {"step1": r"C:\step1\imgB.png", "step2": r"C:\step2\imgB.png"}

    :param sources: a dictionary of (name, path) items,
    where path indicates a folder to search and name the identifier to associate with that path
    :param ext: (optional) extension of items to search
    :param prefix: a prefix that identifies search items
    :param tags: whether to parse tags, these allow multiple entries per source and are encoded in the last split of '-',
    e.g., 'imgA_mask.nrrd' has a tag 'mask', a named result could be 'step1.mask'
    :param parse_tags: (optional) method to parse a filename, returning name and key for the entry, this overrules the default parsing of tags (argument tags)
    :return: the method items, each a dictionary of named paths
    """
    if 'name' in sources.keys():
        raise RuntimeError("'name' is not a valid source name!")
    if parse_fn is None and tags:
        parse_fn = _default_parse_name_tags
    pats = defaultdict(dict)
    for source, path in sources.items():
        input_format = "*"
        if prefix is not None:
            input_format = prefix + input_format
        if ext is not None:
            input_format += ".{}".format(ext)
        input_fn = input_format.format(prefix, ext)
        input_pattern = os.path.join(path, input_fn)
        for pat_path in sorted(glob.glob(input_pattern)):
            pat_name = os.path.basename(pat_path)
            pat_name = os.path.splitext(pat_name)[0]
            source_key = source
            if parse_fn is not None:
                pat_name, source_key = parse_fn(pat_name, source_key)

            pats[pat_name]['name'] = pat_name
            pats[pat_name][source_key] = pat_path

    for pat in OrderedDict(pats).values():
        yield pat