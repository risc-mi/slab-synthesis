#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import argparse
import os

_config = None


def get_config():
    global _config
    if _config is None:
        config = dict()

        # source folder containing the decathlon dataset, the .tar files for each task
        config['source'] = r'<path-to-dataset>'

        # root folder for any output
        config['root'] = r'<path-to-results>'

        # planar size N (int) of NxNxM slabs
        config['size'] = 192

        # whether to shuffle samples or to iterate over the entire dataset
        config['shuffle'] = False

        # number of samples to write in the export script
        config['sample_count'] = None

        # prefix to used for grouping the exported samples, different sample sets are evaluated separately during evaluation
        config['prefix'] = 'ds'

        # during import and preprocessing, individual tasks may be skipped
        # this does however not exclude them from sampling
        config['skip'] = []

        # whether to override existing data during import and preprocessing
        config['override'] = False

        # base spacing (in mm) to use
        # note that scaling factors are defined for individual tasks (see pipeline.get_settings) that relate to this
        # absolute base spacing, i.e., Task 2 (Hearth) uses a factor of 1.0 (100%), Task 1 (BRATS) a factor of 0.5 (50%)
        config['spacing'] = (2, 2, 2)

        # slice number M (int) of NxNxM slabs
        config['slices'] = 64

        # split (float), i.e., the percentage (0.0-1.0) of samples to use for training,
        # the remainder is assigned to the testing split
        config['split'] = 0.8

        # downsample factor (int) to use for segmentations, i.e., reduces the volume size by a factor 2^d
        # this corresponds to the int_downsize used within voxelmorph and only affects
        # sample generation during training, specifically the binary contour masks
        config['int_downsize'] = 2

        # range (min, max) of overlapping slices between the slabs generated for a sample
        # the overlap for any given sample is randomly chosen within this range
        config['overlap_range'] = (4, 24)

        # range (min, max) of slices to shift the overlap center from the sample middle
        # the shift for any given sample is randomly chosen within this range
        config['shift_range'] = (-8, 9)

        # range (min, max) of slices to translate one of the resulting slabs from its original position
        # the translation for any given sample is randomly chosen within this range
        config['trans_range'] = (-2, 3)

        # list of decathlon tasks (integer ids) to sample from
        # if None, all tasks will be sampled
        config['tasks'] = None

        # whether to compress images to safe storage
        config['compress'] = True

        # the default extension to use for output image files, must be supported by SimpleITK
        # supported formats are, e.g., nrrd, nii, nii.gz, mha, mhd
        config['ext'] = 'nrrd'

        # whether to evaluate image (intensity-based) metrics
        config['image_metrics'] = False

        # whether to evaluate label (overlap) metrics
        config['label_metrics'] = True

        # list of models to evaluate
        config['models'] = ['dummy1', 'dummy2']

        # metric to export in the final score tables
        config['export_metric'] = 'dsc'

        # whether to export plots
        config['export_plots'] = True

        _config = config

    return argparse.Namespace(**_config)


def _get_root():
    config = get_config()
    root = getattr(config, 'root', None)
    if not root or '<' in root:
        raise RuntimeError("Please check the configuration before running any scripts!"
                           "The attribute 'root' has not been set - see run/config.py and the documentation in README.md for further information.")
    return root


def get_result_root():
    p = os.path.join(_get_root(), 'results')
    return p


def get_resource_root():
    p = os.path.join(_get_root(), 'resources', 'decathlon')
    return p


def get_local_dataset_root():
    config = get_config()
    source = getattr(config, 'source', None)
    if not source or '<' in source:
        raise RuntimeError("Please check the configuration before running any scripts!"
                           "The attribute 'source' has not been set - see run/config.py and the documentation in README.md for further information.")
    return source


def get_default_extension():
    ext = get_config().ext
    return ext


def get_compression_enabled():
    ext = get_config().compress
    return ext
