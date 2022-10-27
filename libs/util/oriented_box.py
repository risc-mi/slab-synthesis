#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import itertools
import traceback
from typing import Union, Optional

import SimpleITK as sitk
import numpy as np
import vtk

from libs.util.file import read_json, write_json
from libs.util.misc import native, ndtuple


class OrientedBox:
    def __init__(self, origin=np.zeros(3), direction=np.eye(3), size=np.zeros(3)):
        self._origin = np.copy(origin)
        self._direction = np.copy(direction)
        self._size = np.copy(size)
        self._observers = set()

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction

    @property
    def size(self):
        return self._size

    @property
    def valid(self):
        return all(s != 0 for s in self._size)

    @origin.setter
    def origin(self, v):
        if not np.array_equal(v, self._origin):
            self._origin = np.copy(v)
            self._notify()

    @direction.setter
    def direction(self, v):
        if not np.array_equal(v, self._direction):
            self._direction = np.copy(v)
            self._notify()

    @size.setter
    def size(self, v):
        if not np.array_equal(v, self._size):
            self._size = np.copy(v)
            self._notify()

    def get_voxel_size(self, spacing):
        if np.isscalar(spacing):
            spacing = [spacing] * 3
        return np.ceil(np.divide(self.size, spacing)).astype(int).tolist()

    def get_voxel_extent(self, spacing):
        return tuple(itertools.chain(*zip([0]*3, np.subtract(self.get_voxel_size(spacing), 1))))

    def adjust(self, ref: Union[str, sitk.Image, vtk.vtkImageData, vtk.vtkPolyData, 'OrientedBox']):
        """
        adjusts the oriented box to the specified reference
        :param ref: an (sitk) image or path to one
        :return:
        """
        if isinstance(ref, str):
            ref = sitk.ReadImage(ref)

        if isinstance(ref, sitk.Image):
            origin = np.asarray(ref.GetOrigin())
            direction = np.reshape(ref.GetDirection(), (3, 3))
            size = np.multiply(ref.GetSize(), ref.GetSpacing())
            self._apply(origin, direction, size)
        elif isinstance(ref, vtk.vtkImageData):
            origin = np.asarray(ref.GetOrigin())
            from libs.util.interop import vtk_to_numpy_matrix
            direction = vtk_to_numpy_matrix(ref.GetDirectionMatrix())
            size = np.multiply(ref.GetDimensions(), ref.GetSpacing())
            self._apply(origin, direction, size)
        elif isinstance(ref, vtk.vtkPolyData):
            bounds = ref.GetBounds()
            self.set_bounds(bounds)
        elif isinstance(ref, OrientedBox):
            origin = ref.origin
            direction = ref.direction
            size = ref.size
            self._apply(origin, direction, size)
        else:
            raise RuntimeError("Not a valid type of reference: {}".format(type(ref).__name__))

    def set_bounds(self, bounds):
        """
        adjusts the oriented box to bounds data as it is commonly used in vtk
        :param bounds: list of bounds, i.e., for each axis min and max: [min_x, max_x, ...]
        """
        end = bounds[1:None:2]
        origin = bounds[0:None:2]
        direction = np.eye(3)
        size = np.subtract(end, origin)
        self._apply(origin, direction, size)

    def standardize(self):
        corners = list(self._get_corners())
        end = np.max(corners, axis=0)
        origin = np.min(corners, axis=0)
        direction = np.eye(3)
        size = np.subtract(end, origin)
        self._apply(origin, direction, self._size)

    def _apply(self, origin, direction, size):
        same = np.array_equal(self._origin, origin) and np.array_equal(self._direction, direction) and np.array_equal(self._size, size)
        if not same:
            self._origin = np.copy(origin)
            self._direction = np.copy(direction)
            self._size = np.copy(size)
            self._notify()

    def _notify(self):
        for o in self._observers:
            try:
                o()
            except Exception as ex:
                traceback.print_exc()
                raise RuntimeError("Failed to notify observers: {}".format(ex))

    def is_within(self, ref) -> bool:
        """
        test whether the specified reference object is within the oriented box
        :param ref: either a coordinate, image (sitk or path), mesh (vtkPolyData) or another OrientedBox
        """
        if isinstance(ref, [sitk.Image, str, vtk.vtkPolyData]):
            roi = OrientedBox()
            roi.adjust(ref)
        elif isinstance(ref, OrientedBox):
            roi = ref
        else:
            try:
                return self._is_point_within(ref)
            except:
                raise RuntimeError("Not a valid type of reference: {}".format(type(ref).__name__))

        return all(self._is_point_within(p) for p in roi._get_corners())

    def _is_point_within(self, point) -> bool:
        """
        test whether a point coordinate is within the oriented box
        """
        if isinstance(point, [list, tuple]):
            point = np.asarray(point)
        elif not isinstance(point, np.ndarray):
            raise RuntimeError("Invalid point type: {}".format(type(point).__name__))
        if point.shape != 3:
            raise RuntimeError("Points must be defined in 3D space!")
        if not np.issubdtype(point.dtype, np.numeric):
            raise RuntimeError("Points be of a numeric dtype!")

        offset = np.subtract(point, self._origin)
        offset = np.matmul(self._direction, offset)
        return all(c >= 0 and c <= s for c, s in zip(offset, self._size))

    def _get_corners(self):
        """
        get all corner positions of the oriented box
        """
        # permute axis activations
        for perm in itertools.product((0, 1), repeat=3):
            size = np.copy(self._size)
            for d, active in enumerate(perm):
                if not active:
                    size[d] = 0
            yield np.add(self._origin, np.matmul(self._direction, size))

    def add_observer(self, observer):
        self._observers.add(observer)

    def remove_observer(self, observer):
        self._observers.discard(observer)

    def write(self, path):
        write_json(self.to_dict(), path)

    def to_matrix(self, spacing: Optional[float] = None) -> np.ndarray:
        """
        Returns a matrix to transform coordinates in within the oriented box to absolute positions
        :param spacing: when passing a (voxel) spacing, the oriented box can be used to transform image coordinates
        into absolute coordinates (ignoring the boxes size), otherwise values within [0.0, 1.0] can be used to transform relative locations within
        the size (i.e., physical extent) of the oriented box.
        :return: absolute (i.e., "physical") position
        """

        spacing = self.size if spacing is None else ndtuple(spacing, 3) + (1.0,)

        mat_scale = np.eye(4)
        np.fill_diagonal(mat_scale, spacing)

        mat_rot = np.eye(4)
        mat_rot[:3, :3] = self.direction

        mat_trans = np.eye(4)
        mat_trans[:3, 3] = self.origin

        # Index to Point Transform: PhysicalPoint = Origin + (Spacing * IdentityMatrix) * (RotationMatrix) * Index
        mat_t = np.matmul(mat_trans, np.matmul(mat_rot, mat_scale))
        return mat_t

    def as_image(self, spacing, pixelID=sitk.sitkUInt8, components=1):
        """
        returns an image filling the OrientedBox
        :param spacing: physical spacing of the image
        :param pixelID: pixel type to use
        """
        size = self.get_voxel_size(spacing)
        img = sitk.Image(size, pixelID, components)
        img.SetOrigin(native(self.origin))
        img.SetDirection(native(self.direction.flatten()))
        img.SetSpacing(native(spacing))
        return img

    def to_dict(self):
        return {
                'origin': native(self._origin),
                'direction': native(self._direction),
                'size': native(self._size)
            }

    @staticmethod
    def load(path):
        data = read_json(path)
        return OrientedBox.from_dict(data)

    @staticmethod
    def from_dict(data):
        origin = np.asarray(data['origin'])
        direction = np.asarray(data['direction'])
        size = np.asarray(data['size'])
        return OrientedBox(origin=origin, direction=direction, size=size)

    @staticmethod
    def from_image(img: sitk.Image):
        b = OrientedBox()
        b.adjust(img)
        return b

    def extract(self, img: sitk.Image, spacing=None, default_value=None, interpolator=sitk.sitkLinear):
        """
        Resamples the image to the ROI defined by the oriented box
        :param img: image to resample
        :param spacing: uniform spacing value or voxel spacing tuple,
        :param default_value: fill value for voxels outside the original image, if not specified the minimum value in img is used
        :param interpolator: interpolator to use
        :return: extracted image
        """
        if default_value is None:
            stats = sitk.MinimumMaximumImageFilter()
            stats.Execute(img)
            default_value = stats.GetMinimum()
        size = native(self.get_voxel_size(spacing))
        return sitk.Resample(img, size,
                sitk.Transform(img.GetDimension(), sitk.sitkIdentity), interpolator,
                native(self.origin), native(spacing),  native(self.direction.flatten()),
                default_value, img.GetPixelIDValue(), False)