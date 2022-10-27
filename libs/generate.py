#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os
import random
import sys
from collections import defaultdict

import SimpleITK as sitk
import numpy as np

from libs.pipeline import get_settings
from libs.samples import get_samples, create_split, enumerate_names
from libs.util.augmenter import Augmenter
from libs.util.file import read_json, write_json
from run.config import get_default_extension
from libs.util.image import make_affine, resize
from libs.util.misc import default
from libs.util.oriented_box import OrientedBox


def synthethic_slabs(input_root, input_size,
                     overlap_range, shift_range, trans_range,
                     downsize=2, register=False, split=None, tasks=None,
                     return_tensors=True, return_split=False, shuffle=None):
    """
    Generator for registration of synthesised slabs. The generator has two modes:
    - train: returns (invols, outvols, masks)
    - registration: returns name, (volA, segA), (volB, segB)

    param input_root: root folder for the input slabs
    param labels
        seg_names: List of corresponding seg files to load, or list of preloaded volumes.
        downsize: downsampling factor for the binary masks.
        register: whether to use the generator in registration or train mode.
        randomize: allows to explicitly enable shuffle of results, by default its False for registration mode and True for train mode
    """
    ext = get_default_extension()
    samples = get_samples(input_root, ext)
    settings = get_settings()
    shuffle = default(shuffle, not register)

    split_path = os.path.join(input_root, 'split.json')
    if not os.path.exists(split_path):
        if split is None:
            raise RuntimeError("No split file available at: {}".format(split_path))
        print("creating a new split ({:.0%} train)".format(split))
        split = create_split(samples, split)
        write_json(split, split_path)
    else:
        print("Using split file: {}".format(split_path))
        split = read_json(split_path)
    split_map = dict((n, s) for s, l in split.items() for n in l)

    def _fix_size(img):
        if not np.array_equal(img.GetSize(), input_size):
            img = sitk.Resample(img, input_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                img.GetOrigin(), img.GetSpacing(), img.GetDirection(),
                                0, sitk.sitkUnknown, True)
        return img

    msg = "Sampling from all tasks" if tasks is None else "Sampling from tasks: {}".format(sorted(tasks))
    print(msg)
    tasks = None if tasks is None else set(tasks)

    names = list(samples.keys()) if register else list(split['train'])
    names = sorted(names)
    task_names = defaultdict(list)
    for name in names:
        sample = samples[name]
        task = name.split('__')[0]
        task_nr = int(task.lstrip('t'))
        if tasks is None or task_nr in tasks:
            task_names[task].append(name)
            if {'image', 'label', 'mask'}.symmetric_difference(sample.keys()):
                raise RuntimeError("Element is incomplete: {}".format(name))
            missing = list(k for k, p in sample.items() if not os.path.exists(p))
            if missing:
                raise RuntimeError("Element has missing files: {}".format(missing))

    axis = -1
    seed = 0
    rand = random.Random(seed)
    rnd = np.random.RandomState(seed)
    augmenter = Augmenter()
    augmenter.interpolator = 'linear'
    converter = SampleConverter(register=register,
                                tensors=return_tensors,
                                downsize=downsize,
                                split=return_split)

    name_iter = enumerate_names(task_names, shuffle=shuffle, group=True, rand=rand)
    for name in name_iter:
        sample_split = split_map.get(name, 'train')
        task = int(name.split('__')[0].lstrip('t'))
        setting = settings[task]
        background = setting['background']
        threshold = setting['threshold']
        scale = setting['scale']

        sample = samples[name]
        img_path = sample['image']
        label_path = sample['label']
        mask_path = sample['mask']

        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.Resample(mask, img, sitk.Transform(), sitk.sitkNearestNeighbor)

        label = None
        label_centroid_slice = None
        if register:
            label = sitk.ReadImage(label_path)
            label = sitk.Resample(label, img, sitk.Transform(), sitk.sitkNearestNeighbor)
            ls_stat = sitk.LabelShapeStatisticsImageFilter()
            ls_stat.ComputeOrientedBoundingBoxOff()
            ls_stat.ComputePerimeterOff()
            ls_stat.ComputeFeretDiameterOff()
            ls_stat.Execute(label != 0)
            label_centroid_slice = label.TransformPhysicalPointToIndex(ls_stat.GetCentroid(1))[axis]


        n_channels = img.GetNumberOfComponentsPerPixel()
        channel_iter = iter([None])
        if n_channels > 1:
            if shuffle:
                channel_iter = iter([random.randint(0, n_channels-1)])
            else:
                channel_iter = list(range(img.GetNumberOfComponentsPerPixel()))

        for channel_index in channel_iter:
            cimg = img if channel_index is None else sitk.VectorIndexSelectionCast(img, channel_index)
            cname = name if channel_index is None else '{}__ch_{:02d}'.format(name, channel_index+1)

            img_size = cimg.GetSize()
            n_input_slices = img_size[axis]
            overlap = rnd.randint(*overlap_range) if not register else overlap_range[1] - 1
            shift = rnd.randint(*shift_range) if not register else 0
            trans = rnd.randint(*trans_range) if not register else 0
            slice_range = (abs(min(0, trans)), n_input_slices-(input_size[axis]+max(0, trans)))
            if np.subtract(*slice_range) >= 0:
                #print("WARNING: sample {} does not have enough slices ({}) to extract an RoI of {}"
                #      "".format(cname, n_input_slices, input_size[axis], overlap), file=sys.stderr)
                idx = (n_input_slices-input_size[axis])//2
                slice_range = (idx, idx+1)
                trans = 0
                shift = 0

            # define the range of suitable slices
            slice_index = (rnd.randint(*slice_range)
                           if not register
                           else (label_centroid_slice - input_size[axis] // 2))
            slice_index = np.clip(slice_index, a_min=slice_range[0], a_max=slice_range[1]).item()

            # adapt overlap to translation
            overlap = overlap_range[0] + max(overlap - overlap_range[0], abs(trans))
            overlap = np.clip(overlap, a_min=overlap_range[0], a_max=overlap_range[1]-1).item()

            roi_index = list((s1-s2)//2 for s1, s2 in zip(img_size, input_size))

            # set up the roi and augmenter
            roi = OrientedBox()
            roi.size = np.multiply(cimg.GetSpacing(), input_size)
            roi.direction = np.reshape(cimg.GetDirection(), (3, 3))
            augmenter.background = background
            augmenter.contrast_threshold = threshold
            augmenter.scale_factor = scale

            # calculate the origin for each slab
            roi_index[axis] = slice_index
            originA = cimg.TransformIndexToPhysicalPoint(roi_index)
            roi_index[axis] = slice_index + trans
            originB = cimg.TransformIndexToPhysicalPoint(roi_index)

            # swap randomly
            originA, originB = rand.sample([originA, originB], 2)

            # extract, leave the first slab unchanged (intensity augmentation still applies)
            augmenter.unset()
            roi.origin = originA
            imgA = augmenter.apply_image(cimg, roi)
            maskA = augmenter.apply_mask(mask, roi)
            labelA = augmenter.apply_mask(label, roi) if register else None

            # in case some scaling messed up
            imgA = _fix_size(imgA)
            maskA = _fix_size(maskA)
            labelA = _fix_size(labelA) if register else None

            # extract, set geometric augmentation for the other
            augmenter.next(cimg)
            roi.origin = originB
            imgB = augmenter.apply_image(cimg, roi)
            maskB = augmenter.apply_mask(mask, roi)
            labelB = augmenter.apply_mask(label, roi) if register else None

            # in case some scaling messed up
            imgB = _fix_size(imgB)
            maskB = _fix_size(maskB)
            labelB = _fix_size(labelB) if register else None

            # calculate the overlap regions
            n_slices = input_size[axis]
            mid_index = n_slices // 2 + shift
            overlap_start = max(0, mid_index - overlap//2)
            overlap_end = min(n_slices, overlap_start + overlap)
            sliceA = [slice(None)] * 3
            sliceA[axis] = slice(overlap_end, None)
            sliceB = [slice(None)] * 3
            sliceB[axis] = slice(0, overlap_start)

            # clear respective halfs
            imgA[sliceA] = background
            maskA[sliceA] = 0
            imgB[sliceB] = background
            maskB[sliceB] = 0

            if register:
                labelA[sliceA] = 0
                labelB[sliceB] = 0

            r_int = (background, 5000)
            imgA = sitk.Clamp(imgA, sitk.sitkUnknown, r_int[0], r_int[1])
            imgB = sitk.Clamp(imgB, sitk.sitkUnknown, r_int[0], r_int[1])

            if trans != 0:
                # in case one image has been shifted, resample to the same physical space
                imgB = sitk.Resample(imgB, imgA, sitk.Transform(), sitk.sitkLinear, background)
                maskB = sitk.Resample(maskB, imgA, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
                labelB = sitk.Resample(labelB, imgA, sitk.Transform(), sitk.sitkNearestNeighbor, 0)

            converter.min_overlap = overlap_range[0]
            try:
                yield converter.convert(cname,
                                        imgA=imgA, imgB=imgB,
                                        maskA=maskA, maskB=maskB,
                                        labelA=labelA, labelB=labelB,
                                        split=sample_split)
            except Exception as ex:
                print(ex, file=sys.stderr)


def _load_vol(img, add_feat_axis=True, add_batch_axis=True, resize_factor=1, ret_affine=True):
    vol = sitk.GetArrayFromImage(img).T.squeeze()
    affine = make_affine(img)

    if np.issubdtype(vol.dtype, np.uint16):
        vol = np.asarray(vol, dtype=np.int16)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol


class SampleConverter:
    def __init__(self, downsize, register=True, tensors=True, min_overlap=None, split=False):
        self.register = register
        self.return_tensors = tensors
        self.return_split = split
        self.downsize = downsize
        self.min_overlap = min_overlap
        self.zeros = None

    def _downsample_seg(self, seg):
        if self.downsize > 1:
            prob_seg = np.zeros((*seg.shape[:4], 1), dtype=seg.dtype)
            prob_seg[0, ..., 0] = seg[0, ..., 0] == 1
            return prob_seg[:, ::self.downsize, ::self.downsize, ::self.downsize, :]
        else:
            return seg

    def convert(self, name, imgA, imgB, maskA, maskB, labelA, labelB, split):
        # load vols and seg
        volA = _load_vol(imgA, ret_affine=False).astype(np.float32) if self.return_tensors else imgA
        volB = _load_vol(imgB, ret_affine=False).astype(np.float32) if self.return_tensors else imgB
        segA = _load_vol(maskA, ret_affine=False).astype(np.bool) if self.return_tensors else maskA
        segB = _load_vol(maskB, ret_affine=False).astype(np.bool) if self.return_tensors else maskB

        if self.register:
            lblA = _load_vol(labelA, ret_affine=False).astype(np.uint8) if self.return_tensors else labelA
            lblB = _load_vol(labelB, ret_affine=False).astype(np.uint8) if self.return_tensors else labelB
            res = [name, (volA, segA, lblA), (volB, segB, lblB)]
            if self.return_split:
                res.append(split)
            return tuple(res)
        else:
            if not self.return_tensors:
                raise RuntimeError("Returning images (tensors=False) is only supported for registration")
            if self.return_split:
                raise RuntimeError("Returning split (split=True) is only supported for registration")

            # extend masks to entire z slice
            sliceA = np.copy(segA)
            sliceB = np.copy(segB)

            lbl_slicesA = np.any(segA, axis=(0, 1, 2, 4))
            lbl_slicesB = np.any(segB, axis=(0, 1, 2, 4))
            if self.min_overlap is not None:
                lbl_and = np.logical_and(lbl_slicesA, lbl_slicesB)
                lbl_count = np.count_nonzero(lbl_and)
                if lbl_count < self.min_overlap:
                    raise RuntimeError("Unsufficient overlap: drawing a new sample!")

            sliceA[:, :, :] = lbl_slicesA[:, np.newaxis]
            sliceB[:, :, :] = lbl_slicesB[:, np.newaxis]

            low_segA = self._downsample_seg(sliceA)
            low_segB = self._downsample_seg(sliceB)

            # cache zeros
            if self.zeros is None:
                ls_stat = volA.shape[1:-1]
                self.zeros = np.zeros((1, *ls_stat, len(ls_stat)), dtype=np.float32)

            invols = [volA, volB]
            outvols = [volB, self.zeros]
            masks = [(segA, sliceA, low_segA), (segB, sliceB, low_segB)]
            return invols, outvols, masks, name