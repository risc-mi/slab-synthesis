#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os

import SimpleITK as sitk
import numpy as np

from libs.samples import get_sample_info
from libs.util.augmenter import generate_perlin_transform
from libs.util.file import mkdirs
from run.config import get_default_extension


def generate_random_warps(sample_root, warps_path,
                          override=True):

    ext = get_default_extension()
    sample_map, sample_sets, sample_meta = get_sample_info(sample_root, ext=ext)
    warp_map, _, _ = get_sample_info(warps_path, ext=ext)

    mkdirs(warps_path)
    rnd = np.random.RandomState(0)
    for sample_idx, (sample_id, sample_items) in enumerate(sample_map.items()):
        warp_result = warp_map.get(sample_id, dict())
        if override or not warp_result:
            print("Processing {}... ".format(sample_id))

            # for testing purposes
            # we simply generate a displacement transform from perlin noise
            maskA = sitk.ReadImage(sample_items['A_mask'], sitk.sitkUInt8)
            st = generate_perlin_transform(maskA, rnd=rnd)
            df = st.GetDisplacementField()

            path = os.path.join(warps_path, '{}__A_warp.{}'.format(sample_id, ext))
            sitk.WriteImage(df, path)