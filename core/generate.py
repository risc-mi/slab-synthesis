#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os
from typing import Optional

import SimpleITK as sitk

from libs.generate import synthethic_slabs
from libs.util.file import mkdirs, write_json, read_json
from run.config import get_default_extension, get_compression_enabled
from libs.util.misc import default


def write_sample(sample, dir: str, prefix: str):
    """
    exports a single sample to the output directory
    :param prefix: prefix to prepend to the filename
    :param sample: sample data to write
    :param dir: directory to write to
    :return:
    """
    ext = get_default_extension()
    compress = get_compression_enabled()
    name, (imgA, maskA, labelA), (imgB, maskB, labelB), split = sample

    print("sampling from {}...".format(name))
    imgA = sitk.Cast(imgA, sitk.sitkInt16)
    imgB = sitk.Cast(imgB, sitk.sitkInt16)
    maskA = sitk.Cast(maskA, sitk.sitkUInt8)
    maskB = sitk.Cast(maskB, sitk.sitkUInt8)
    labelA = sitk.Cast(labelA, sitk.sitkUInt8)
    labelB = sitk.Cast(labelB, sitk.sitkUInt8)

    export_items = {
        'A': imgA,
        'A_mask': maskA,
        'A_label': labelA,
        'B': imgB,
        'B_mask': maskB,
        'B_label': labelB
    }
    sample_name = '{}-{}'.format(prefix, name) if prefix is not None else name
    for item_key, item in export_items.items():
        path = os.path.join(dir, '{}__{}.{}'.format(sample_name, item_key, ext))
        sitk.WriteImage(item, path, compress)
    return sample_name, split


def get_generator(source_path, config):
    """
    Configures the generator for synthetic slabs for exporting samples.
    The following attributes are used from the configuration:
    - size: size N of the resulting slab volumes (NxNxM)
    - slices: size M of the resulting slab volumes (NxNxM)
    - overlap_range: (min, max) range of slices the resulting slabs overlap
    - shift_range: (min, max) range of slices to shift the overlap center between the slabs by
    - trans_range: (min, max) range of slices to translate one of the slabs by
    - int_downsize: factor to downsample binary masks by (training only)
    - split: split factor, i.e., percentage of samples used for training
    - tasks: list of tasks to sample from
    - register: whether to create samples for registration/prediction or training, which affects the returned
    contents (samples w./w.o. ground-truth) and returned types (images vs tensors) and
    :param source_path: folder containing the preprocessed dataset to sample from
    :param config: sampling configuration
    :return: configured sample generator
    """
    size = config.size
    input_size = (size, size, config.slices)
    return synthethic_slabs(source_path,
                            input_size=input_size,
                            overlap_range=config.overlap_range,
                            shift_range=config.shift_range,
                            trans_range=config.trans_range,
                            downsize=config.int_downsize,
                            split=config.split,
                            tasks=config.tasks,
                            shuffle=config.shuffle,
                            register=True,
                            return_split=True,
                            return_tensors=False)


def generate_samples(gen, count: Optional[int], output_dir: str, prefix: str= 'ds'):
    """
    exports samples generated by the specified generator
    :param gen: generator to sample from
    :param count: number of samples to export
    :param output_dir: directory to same samples to
    :param prefix: prefix to identify the exported sample set in case multiple are used
    """
    mkdirs(output_dir)

    split_path = os.path.join(output_dir, 'split.json')
    print("Using split file: {}".format(split_path))
    split = read_json(split_path) if os.path.exists(split_path) else dict()

    update_rate = 10
    idx = 0
    while True:
        if default(count, -1) > 0 and idx >= count:
            break
        idx += 1

        sample_prefix = '{}{:04d}'.format(prefix, idx)
        sample = next(gen, None)
        if sample is None:
            break

        sample_name, sample_split = write_sample(sample, dir=output_dir, prefix=sample_prefix)

        split.setdefault(sample_split, []).append(sample_name)
        if idx % update_rate == 0:
            # update the split json file every N samples
            write_json(split, split_path)