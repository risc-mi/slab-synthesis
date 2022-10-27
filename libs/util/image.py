#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import SimpleITK as sitk
import numpy as np
import scipy

from libs.util.misc import native, as_tuple, default


def get_center(img: sitk.Image):
    """
    returns the center of an image as an absolute position
    """
    center = get_center_index(img, round=False)
    return img.TransformContinuousIndexToPhysicalPoint(center)


def get_center_index(img: sitk.Image, round=True):
    """
    returns the center of an image as an index
    :param round: whether to round the index to an integer
    """
    c = np.divide(img.GetSize(), 2)
    if round:
        c = np.round(c).astype(np.int)
    return native(c)


def resample(img, spacing, labels=False, size=None, interpolation=None, center=None, center_position=None, default_value=0, extrapolate=False,
             setCallback=None) -> sitk.Image:
    """
    Resamples an image to the specified spacing (in mm)
    Sometimes, it is necessary to enforce an output image size: in this case, size and center can be specified to automatically
    crop the image in the resampling step - this is more effective and accurate than cropping in a postprocessing-step).
    :param img: image to resample
    :param spacing: spacing in mm to target
    :param labels: whether the image contains labels (mask or segmentation), this affects the default interpolation
    :param size: (optional) size of the output image to enforce (relative to the center), if not specified, the size is calculated
    :param interpolation: (optional) SimpleITK interpolation method, if not specified, the default is determined from the image
    :param center: center index of the image, if not specified, the center will not change
    :param center_position: center position of the image, alternative to the argument center (which is an index)
    :param default_value: fill value for sampled areas outside the original extent
    :param extrapolate: whether to enable extrapolation (outside the volume)
    :return: resampled image
    """
    filter = resample_filter(img=img, spacing=spacing, labels=labels, size=size, interpolation=interpolation, center=center, center_position=center_position,
                             default_value=default_value, extrapolate=extrapolate)
    if filter is None:
        return img
    if setCallback is not None:
        setCallback(filter)
    return filter.Execute(img)


def convert_img_dims(ref: sitk.Image, v):
    """
    for a value that can be a tuple or scalar, return a spacing tuple of the reference image dimension
    """
    return [v] * ref.GetDimension() if np.isscalar(v) else v


def resample_filter(img, spacing, labels=False, size=None, interpolation=None, center=None, center_position=None, default_value=0, extrapolate=False):
    """
    Instanciates the SimpleITK filter based on the specified image
    For the arguments see resample
    """
    spacing = convert_img_dims(img, spacing)
    old_spacing = img.GetSpacing()
    new_spacing = [v for v in spacing]
    old_size = img.GetSize()
    if size is None:
        size = [int(0.5 + old_size[i] * s / new_spacing[i]) for i, s in enumerate(old_spacing)]
    if center is None:
        if center_position is None:
            center = np.multiply(old_size, 0.5)
    else:
        if center_position is not None:
            raise RuntimeError("Either center or center_position may be specified - not both!")
    if center_position is None:
        center_position = img.TransformIndexToPhysicalPoint(np.asarray(center, dtype=int).tolist())

    ref = sitk.Image(native(size), img.GetPixelID())
    ref.SetDirection(img.GetDirection())
    ref.SetSpacing(native(new_spacing))

    new_diff = np.subtract(ref.TransformIndexToPhysicalPoint(np.multiply(size, 0.5).astype(int).tolist()), ref.GetOrigin())
    new_origin = center_position-new_diff
    ref.SetOrigin(new_origin)

    trans = sitk.Transform()
    trans.SetIdentity()
    if interpolation is None:
        interpolation = sitk.sitkBSpline if not labels else sitk.sitkNearestNeighbor

    changed = not np.array_equal(spacing, old_spacing)
    if not changed:
        changed = ref.GetSize() != img.GetSize() or ref.GetOrigin() != img.GetOrigin()
    if changed:
        filter = sitk.ResampleImageFilter()
        filter.SetReferenceImage(ref)
        filter.SetTransform(trans)
        filter.SetOutputPixelType(img.GetPixelID() if not labels else sitk.sitkUInt8)
        filter.SetInterpolator(interpolation)
        filter.SetDefaultPixelValue(default_value)
        filter.SetUseNearestNeighborExtrapolator(extrapolate)
        return filter
    else:
        return None


def make_affine(img):
    # get affine transform in LPS
    c = [img.TransformContinuousIndexToPhysicalPoint(p)
         for p in ((1, 0, 0),
                   (0, 1, 0),
                   (0, 0, 1),
                   (0, 0, 0))]
    c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if not batch_axis:
            dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def mask_contour(img: sitk.Image, threshold=0, axes=None, radius=2, island=True):
    """
    Simple method to mask the contour of an object.
    The script first creates an initial mask through thresholding,
    applies a binary closing operation and and then fills holes on slices
    extracted from the specified axes. Finally the method applies a binary opening operation.
    :param img: image to mask
    :param threshold: intial threshold
    :param axes: list of axes to fill holes, defaults to axial (-1)
    :param radius: radius for the open/close operations, 0 disables the operations
    :param island: whether to mask the largest remaining island
    :return: the resulting mask
    """
    dims = img.GetDimension()
    mask = img > threshold
    if radius > 0:
        mask = sitk.BinaryMorphologicalClosing(mask, [radius]*3)
    if axes is None:
        axes = [-1]

    # traverse the axes
    for axis in axes:
        # fill holes in every slice
        indices = [slice(None)] * dims
        for i in range(mask.GetSize()[axis]):
            indices[axis] = i
            sliced = mask[tuple(indices)]
            sliced = sitk.BinaryFillhole(sliced)
            mask[tuple(indices)] = sliced

    if radius > 0:
        mask = sitk.BinaryMorphologicalOpening(mask, [radius]*dims)
    return largest_island(mask) if island else mask


def largest_island(mask: sitk.Image) -> sitk.Image:
    """
    returns a mask with only the largest island (i.e., connected component)
    :param mask: original mask to filter
    :return: mask of the largest island
    """
    return filter_islands(mask, n_largest=1)


def get_roi(mask: sitk.Image):
    """
    returns the region of interest defined by the mask.
    The region of interest is a bounding box containing the labelled areas within
    the mask.
    :param mask: mask to base the region of interest on
    :return: tuple (index, size) of the region of interest
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.ComputePerimeterOff()
    stats.ComputeFeretDiameterOff()
    stats.ComputeOrientedBoundingBoxOff()
    stats.Execute(mask)
    bounds = stats.GetBoundingBox(stats.GetLabels()[-1])

    index = bounds[:3]
    size = bounds[3:]
    return index, size


def match_image_information(img1: sitk.Image, img2: sitk.Image, match_origin=True, match_spacing=True, match_orientation=True):
    """
    returns true when both images have the same geometrical information
    :param match_coordinates: whether to consider or ignore the origin
    :param match_spacing: whether to consider or ignore the spacing
    """
    if img1.GetDimension() != img2.GetDimension():
        return False
    if img1.GetSize() != img2.GetSize():
        return False
    if match_origin and (img1.GetSize() != img2.GetSize()):
        return False
    if match_spacing and (img1.GetSpacing() != img2.GetSpacing()):
        return False
    if match_orientation and (img1.GetDirection() != img2.GetDirection()):
        return False
    return True


def crop_roi(volume: sitk.Image, mask: sitk.Image, padding=None, axes=None):
    """
    crops the volume and mask to the region of interest defined by the mask.
    The region of interest is a bounding box containing the labelled areas within
    the mask. When providing a padding value, the region of interest will be padded
    by the specified value interpreted as a physical unit, i.e., millimeter.
    :param volume: volume to crop
    :param mask: mask to base the region of interest on and also crop
    :param padding: padding in millimeter
    :param axes: list of axes to restrict the crop operation to, if None, all axes are affected
    :return: the cropped volume and mask
    """

    if not match_image_information(volume, mask):
        mask = sitk.Resample(mask, volume, sitk.Transform(), sitk.sitkNearestNeighbor)

    index, size = get_roi(mask)
    if padding is not None:
        if np.isscalar(padding):
            padding = np.repeat(padding, volume.GetDimension())
        padding = np.divide(padding, volume.GetSpacing())
        padding = np.ceil(padding)

        end = np.min((np.add(size, index) + padding, volume.GetSize()), axis=0)
        index = np.max((index-padding, np.zeros(padding.size)), axis=0)
        size = end - index
        index = index.astype(int).tolist()
        size = size.astype(int).tolist()

    if axes is not None:
        naxes = set(range(3)).difference(list(range(3))[a] for a in as_tuple(axes))
        index = list(index)
        size = list(size)
        for axis in naxes:
            index[axis] = 0
            size[axis] = volume.GetSize()[axis]


    roi_volume = sitk.RegionOfInterest(volume, size, index)
    roi_mask = sitk.RegionOfInterest(mask, size, index)
    roi_mask.CopyInformation(roi_volume)
    return roi_volume, roi_mask


def filter_islands(mask: sitk.Image, n_largest=None, min_voxels=None, min_mm3=None) -> sitk.Image:
    """
    filters connected islands in the mask
    :param mask: input mask to filter
    :param n_largest: when specified, filters the first n largest islands
    :param min_voxels: when specified, filters islands given a minimum number of voxels
    :param min_mm3: when specified, filters islands given a minimum number of mm3 volume
    :return:
    """
    if min_mm3 is not None:
        mm3_voxels = np.divide(min_mm3, np.prod(mask.GetSpacing()))
        min_voxels = max(mm3_voxels, default(min_voxels, 0))

    filter_components = sitk.ConnectedComponentImageFilter()
    filter_stats = sitk.LabelShapeStatisticsImageFilter()
    components = filter_components.Execute(mask)

    filter_stats.ComputeFeretDiameterOff()
    filter_stats.ComputeOrientedBoundingBoxOff()
    filter_stats.ComputePerimeterOff()
    filter_stats.Execute(components)
    islands = sorted(list((l, filter_stats.GetNumberOfPixels(l)) for l in filter_stats.GetLabels() if l != 0),
                 key=lambda t: t[1], reverse=True)

    if min_voxels is not None:
        islands = list(filter(lambda r: r[1] >= min_voxels, islands))
    if n_largest is not None:
        islands = islands[:n_largest]

    res = None
    for island in islands:
        island_mask = components == island[0]
        res = island_mask if res is None else sitk.Or(res, island_mask)
    if res is None:
        res = fill_mask(mask, 0)
    return res


def fill_mask(ref: sitk.Image, value: int=1, type=sitk.sitkUInt8):
    """
    creates a mask with the same geometry and proportion of the specified image and fills it with the specified value
    :param image: reference image for geometry and size of the mask
    :param value: value to fill into the mask
    """
    arr = np.full(np.transpose(ref.GetSize()), value, dtype=np.uint8)
    mask = sitk.GetImageFromArray(arr.T)
    mask.CopyInformation(ref)
    if type != sitk.sitkUnknown:
        curr_type = mask.GetPixelIDValue()
        if curr_type != type:
            mask = sitk.Cast(mask, type)
    return mask


def mask_nonempty_slices(mask: sitk.Image, axis=-1):
    """
    Masks all slices along axis that do contain any nonzero elements.
    Returns a mask of equal size and orientation as the input mask.
    """
    axes = list(range(mask.GetDimension()))
    targets = tuple(a for a in axes if a != axes[axis])
    slices = np.any(sitk.GetArrayFromImage(mask).T, axis=targets)
    arr = np.zeros(mask.GetSize())
    arr = np.moveaxis(arr, axis, -1)
    arr[..., :] = slices
    arr = np.moveaxis(arr, -1, axis)
    res = sitk.GetImageFromArray(arr.astype(np.uint8).T)
    res.CopyInformation(mask)
    return res