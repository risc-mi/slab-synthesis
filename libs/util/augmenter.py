#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import glob
import os
from typing import Optional

import SimpleITK as sitk
import numpy as np

from libs.util.file import mkdirs
from run.config import get_result_root, get_default_extension
from libs.util.image import get_center, resample
from libs.util.misc import headline
from libs.util.perlin import perlin3d
from libs.util.transform import Transform


def _randint(rnd):
    return rnd.randint(0, np.iinfo(int).max, 1).item()


def _resize(arr, shape):
    img_scale = sitk.GetImageFromArray(arr.T)
    img_scale = resample(img_scale, np.divide(arr.shape, shape),
                         size=shape, interpolation=sitk.sitkLinear, extrapolate=True)
    res = sitk.GetArrayFromImage(img_scale).T
    return res


def _calc_perlin_size(size):
    """
    calculates the closest power of two dimensions for the perlin image
    """
    return tuple(np.power(2, np.round(np.log2(size))).astype(int))


def _get_perlin(ref_shape, out_shape, freq, rnd):
    perlin_size = _calc_perlin_size(ref_shape)
    arr = perlin3d(perlin_size, [freq] * 3, seed=_randint(rnd))
    arr = _resize(arr, out_shape)
    return arr


def generate_perlin_transform(ref: sitk.Image,
                              axes: Optional[set] = None,
                              scale: float = 20, freq=2, grid_size: int = 10,
                              rnd = None):
    """
    generates a displacement field transform from perlin noise
    :param ref: reference image for the shape of the displacement field
    :param grid_size: size of the perlin grid, higher values result in a lower resolution, thus more performance
    :param axes: set of axis indices to apply deformation along, if None, all axes
    :param scale: deformation scale
    :param freq: frequency for the perlin noise function
    :return: displacement field transform
    """
    rnd = np.random.RandomState() if rnd is None else rnd
    extent = np.multiply(ref.GetSize(), ref.GetSpacing())
    grid_size = np.floor_divide(extent, grid_size).astype(int)
    grid_size = np.max((grid_size, [16] * 3), axis=0)
    grid_spacing = np.divide(extent, grid_size)

    grid = list()
    for i in range(3):
        if axes is None or i in axes:
            arr = _get_perlin(grid_size, grid_size, freq=freq, rnd=rnd)
        else:
            arr = np.zeros(grid_size, dtype=np.float64)
        grid.append(arr)

    deform_scale = list((scale if (axes is None or i in axes) else 0)
                        for i in range(3))
    grid = np.asarray(grid, dtype=np.float64).T
    grid = np.multiply(deform_scale, grid)

    df = sitk.GetImageFromArray(grid, True)
    df.SetSpacing(grid_spacing)
    df.SetDirection(ref.GetDirection())
    df.SetOrigin(ref.GetOrigin())
    st = sitk.DisplacementFieldTransform(df)
    return st


class Augmenter:
    def __init__(self):
        self.slicing_axis = -1
        self.rotation_angle = 3
        self.scales = [(0.97, 1.03), (0.97, 1.03), (1, 1)]
        self.background = 0
        self.scale_factor = 1.0
        self.interpolator = 'linear'
        self.deform_scale = 20
        self.deform_perlin = 2

        self.contrast_perlin = 1
        self.contrast_perlin_scale = 0.1
        self.contrast_threshold = 100
        self.contrast_scale = 0.2

        self._deformation = None
        self._transform = Transform.fromSimpleITK(sitk.Transform())
        self._rnd = np.random.RandomState(0)

    def _rand(self, shape=None):
        """
        returns a random value between -1.0 and 1.0
        """
        return self._rnd.random() * 2 - 1.0 if shape is None else self._rnd.random(shape)

    @property
    def deformation(self):
        return self._deformation

    def next(self, image):
        axis = list(range(3))[self.slicing_axis]
        m = np.asarray(image.GetDirection()).reshape([3] * 2)

        vr = sitk.ScaleVersor3DTransform()
        angle = self.rotation_angle * self._rand()
        vr.SetRotation(m[:, axis], np.deg2rad(angle))
        vr.SetCenter(get_center(image))

        vs = sitk.ScaleVersor3DTransform()
        scale = tuple((self._rnd.uniform(*s)) for s in self.scales)
        vs.SetScale(scale)
        center_index = tuple(self._rnd.randint(0, image.GetSize()[d]) for d in range(3))
        vs.SetCenter(image.TransformIndexToPhysicalPoint(center_index))

        axes = set(i for i in range(3) if i != axis)
        scale = self.deform_scale * self.scale_factor
        st = generate_perlin_transform(image, axes, scale=scale, freq=self.deform_perlin, rnd=self._rnd)

        t = sitk.CompositeTransform(3)
        t.AddTransform(st)
        t.AddTransform(vr)
        t.AddTransform(vs)

        self._deformation = st.GetDisplacementField()
        self._transform = Transform.fromSimpleITK(t)

    def unset(self):
        """
        disables geometric augmentation by setting an identity transform
        """
        self._deformation = None
        self._transform = Transform.identity()

    def apply_image(self, image: sitk.Image, ref=None):
        # contrast augmentation for intensities above the threshold
        arr = sitk.GetArrayFromImage(image)
        mask = arr > self.contrast_threshold

        # linear intensity scaling
        scale = 1.00 + self._rand() * self.contrast_scale
        arr[mask] = ((arr[mask] - self.contrast_threshold) * scale) + self.contrast_threshold

        # perlin intensity scaling
        extent = np.multiply(image.GetSize(), image.GetSpacing())
        grid_size = np.floor_divide(extent, 10 / self.scale_factor).astype(int)
        grid_size = np.max((grid_size, [16] * 3), axis=0)
        scale = _get_perlin(grid_size, mask.shape, freq=self.contrast_perlin, rnd=self._rnd)[mask]
        scale = (((scale - scale.min()) / (scale.max() - scale.min())) * 2 * self.contrast_perlin_scale) + (
                    1.00 - self.contrast_perlin_scale)
        arr[mask] = ((arr[mask] - self.contrast_threshold) * scale) + self.contrast_threshold

        h = sitk.GetImageFromArray(arr)
        h.CopyInformation(image)
        image = h
        ref = image if ref is None else ref
        return self._transform.transform_image(image, ref, default_value=self.background,
                                               interpolator=self.interpolator)

    def apply_mask(self, image: sitk.Image, ref=None):
        ref = image if ref is None else ref
        return self._transform.transform_segmentation(image, ref, default_value=0)


def main():
    size = 192
    background = 0
    results_root = get_result_root()
    working_root = os.path.join(results_root, "size{}".format(size))
    input_path = os.path.join(working_root, 'pp')
    output_root = os.path.join(working_root, 'test_augment')

    mkdirs(output_root)
    headline('Vizard: test script for slab augmentation')
    print("The working root path is set to: {}".format(working_root))
    print("Samples are loaded from: {}".format(input_path))
    print("The output directory is set to: {}".format(output_root))

    aug = Augmenter()
    aug.background = background

    ext = get_default_extension()
    input_pattern = os.path.join(input_path, '*.{}'.format(ext))
    data = dict((os.path.basename(fn).split('.')[0], fn)
                for fn in glob.glob(input_pattern))

    for name, fn in data.items():
        print("Processing {}...".format(name))
        slab = sitk.ReadImage(fn)

        aug.next(slab)
        slab = aug.apply_image(slab) if slab.GetPixelID() != sitk.sitkUInt8 else aug.apply_mask(slab)

        ofn = os.path.join(output_root, '{}.{}'.format(name, ext))
        sitk.WriteImage(slab, ofn)

        ofn = os.path.join(output_root, '{}_warp.{}'.format(name, 'mha'))
        sitk.WriteImage(aug._deformation, ofn)


if __name__ == '__main__':
    main()
