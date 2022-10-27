#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import glob
import os
from collections import defaultdict

import SimpleITK as sitk

from run.config import get_default_extension, get_compression_enabled
from libs.util.image import resample, mask_contour, crop_roi, fill_mask


def preprocess_all(input_path, output_path, settings, override, skip):
    os.makedirs(output_path, exist_ok=True)

    ext = get_default_extension()
    compress = get_compression_enabled()
    input_pattern = os.path.join(input_path, '*.{}'.format(ext))
    elements = defaultdict(dict)
    for fn in glob.glob(input_pattern):
        name = os.path.basename(fn).split('.')[0]
        parts = name.split('__')
        element_type = parts[-1]
        element_name = '__'.join(parts[:-1])
        elements[element_name][element_type] = fn
    for name, element in elements.items():
        task, target = name.split('__')
        target, nr = target.split('_')
        task = int(task.lstrip('t'))
        nr = int(nr)
        element['task'] = task
        element['target'] = target
        element['nr'] = nr

    for name, element in elements.items():
        msg = "Processing {}...".format(name)
        print(msg, end='')

        task = element['task']
        if task not in skip:
            image_fn = element['image']
            label_fn = element['label']
            setting = settings[task]
            spacing = setting['spacing']
            background = setting['background']

            image_ofn = os.path.join(output_path, '{}__image.{}'.format(name, ext))
            label_ofn = os.path.join(output_path, '{}__label.{}'.format(name, ext))
            mask_ofn = os.path.join(output_path, '{}__mask.{}'.format(name, ext))
            if override or not all(os.path.exists(p) for p in [image_ofn, label_fn, mask_ofn]):
                image = sitk.ReadImage(image_fn)
                label = sitk.ReadImage(label_fn)
                if image.GetDimension() > 3:
                    c_image = image[..., 0]
                    image = sitk.Compose(list(image[..., i] for i in range(image.GetSize()[-1])))
                else:
                    c_image = image
                image = sitk.DICOMOrient(image)
                label = sitk.DICOMOrient(label)
                c_image = sitk.DICOMOrient(c_image)

                masking = setting['masking'].lower()
                if masking == 'threshold':
                    t = setting['threshold']
                    mask = c_image > t
                elif masking == 'contour':
                    t = setting['threshold']
                    r = setting['radius']
                    mask = mask_contour(c_image, threshold=t, radius=r, axes=[-1], island=True)
                elif masking == 'none':
                    mask = fill_mask(c_image)
                else:
                    raise RuntimeError("Unknown masking method: {}".format(masking))

                image, _ = crop_roi(image, mask)
                label, mask = crop_roi(label, mask)

                size = None
                pp_image = resample(image, spacing, False, size=size, default_value=background)
                pp_label = resample(label, spacing, True, size=size, default_value=0)
                pp_mask = resample(mask, spacing, True, size=size, default_value=0)

                sitk.WriteImage(pp_image, image_ofn, compress)
                sitk.WriteImage(pp_label, label_ofn, compress)
                sitk.WriteImage(pp_mask, mask_ofn, compress)

            print('\r{} DONE'.format(msg))
        else:
            print('\r{} SKIPPED'.format(msg))