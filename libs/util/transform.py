#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from typing import Optional, Union, List

import SimpleITK as sitk
import numpy as np
import vtk

from libs.util.interop import get_interpolator
from libs.util.interop import sitk_to_vtk_image, vtk_to_sitk_image
from libs.util.interop import sitk_to_vtk_transform, vtk_to_sitk_transform
from libs.util.misc import native, ndtuple
from libs.util.oriented_box import OrientedBox


class Transform:
    def __init__(self, inner: Optional[Union[vtk.vtkAbstractTransform, sitk.Transform]], inverse: Optional['Transform'] = None):
        if inner is None and inverse is None:
            raise RuntimeError("Either the inner (sitk) transform or an inverse transform must be specified!")
        if isinstance(inner, sitk.Transform):
            inner = sitk_to_vtk_transform(inner)
        self._inner = inner
        self._displacement_field = None
        self._displacement_spacing = 1.0
        self._inverse = inverse
        self._reference = OrientedBox()
        self._reference.add_observer(self._clear_displacement_field)

    def _inner_as_sitk(self):
        if self._inner is not None:
            return vtk_to_sitk_transform(self._inner)

    def toSimpleITK(self):
        return self._inner_as_sitk()

    @staticmethod
    def identity():
        return Transform(vtk.vtkIdentityTransform())

    @staticmethod
    def fromSimpleITK(t: sitk.Transform, inverse=None) -> 'Transform':
        t = sitk_to_vtk_transform(t)
        return Transform(t, inverse=inverse)

    @staticmethod
    def chain(t1: 'Transform', t2: 'Transform'):
        return Transform.chain_list([t1, t2])

    @staticmethod
    def chain_list(t_list: List['Transform']):
        t = vtk.vtkGeneralTransform()
        for t_elem in reversed(t_list):
            t.Concatenate(t_elem.inner)
        return Transform(t)

    @property
    def reference(self) -> OrientedBox:
        """
        reference space to translate to
        """
        return self._reference

    @property
    def displacement_spacing(self):
        return self._displacement_spacing

    @displacement_spacing.setter
    def displacement_spacing(self, v):
        if v != self._displacement_spacing:
            self._displacement_spacing = v
            self._clear_displacement_field()

    @property
    def displacement_field(self) -> sitk.Image:
        if self._displacement_field is None:
            self._displacement_field = self._calculate_displacement_field()
        return self._displacement_field

    @property
    def inner(self) -> vtk.vtkAbstractTransform:
        if self._inner is None:
            # invert the transform, which should be just a flag in VTK
            return self.inverse._inner.GetInverse()
        return self._inner

    def _get_displacement_spacing(self):
        return ndtuple(self._displacement_spacing)

    @property
    def inverse(self):
        """
        Returns an inverse for the transform.
        """
        if self._inverse is None:
            inverse = Transform(None, inverse=self)
            self._set_inverse(inverse)
        return self._inverse

    def _set_inverse(self, inverse: 'Transform'):
        self._inverse = inverse

    def _clear_displacement_field(self):
        self._displacement_field = None

    def transform_mesh(self, mesh: vtk.vtkPolyData, method='vtk'):
        if method == 'direct':
            t = self.inner
            ps = mesh.GetPoints()
            points1 = list(ps.GetPoint(pid) for pid in range(ps.GetNumberOfPoints()))
            points2 = list(t.TransformPoint(p) for p in points1)
            for id, p in enumerate(points2):
                ps.SetPoint(id, p)

            return mesh
        elif method == 'vtk':
            filter = vtk.vtkTransformPolyDataFilter()
            filter.SetTransform(self.inner)
            filter.SetInputData(mesh)
            filter.Update()
            return filter.GetOutput()
        else:
            raise RuntimeError("Unknown method to transform mesh: {}".format(method))

    def transform_image(self,
                        image: Union[str, vtk.vtkImageData, sitk.Image],
                        ref: Optional[Union[str, sitk.Image, OrientedBox]],
                        method='vtk',
                        interpolator='linear',
                        spacing=None,
                        default_value=0):
        """
        transforms an image
        The method is implemented for SimpleITK and VTK, whereas the SimpleITK method might have to
        resort to less accurate displacement fields to transform the image.
        If a reference is specified, it is used to define the output images extents, otherwise the reference
        associated with the transform is used.
        The spacing of the output image can be specified directly, otherwise it will be extracted from the
        reference (if it is an image) or set to the input image spacing.
        :param image: image to transform; either a path to an image, a vtk or simpleitk image
        :param ref: (optional) reference for the output image; a path, an simpleitk image or an oriented box
        :param method: either 'sitk' or 'vtk'
        :param interpolator: Interpolator to use: see get_interpolator, e.g. 'nearest', 'linear'
        :param spacing: (optional) spacing; if not specified it will be extracted to the reference (if it is an image)
        or default to the input image spacing.
        :param default_value: default value for voxels outside the input image
        :return: transformed image
        """

        if isinstance(image, str):
            image = sitk.ReadImage(image)

        # process the reference
        if isinstance(ref, str):
            ref = sitk.ReadImage(ref)
        ref_roi = OrientedBox()
        ref_spacing = image.GetSpacing()
        if isinstance(ref, (sitk.Image, vtk.vtkImageData)):
            ref_roi.adjust(ref)
            ref_spacing = ref.GetSpacing()
        elif isinstance(ref, OrientedBox):
            ref_roi = ref
        elif ref is None:
            ref_roi = self.reference
        else:
            raise RuntimeError("Invalid reference type: {}".format(type(ref).__name__))

        # if a spacing was specified, override
        spacing = spacing if spacing is not None else ref_spacing

        if method == 'vtk':
            return self._transform_image_vtk(image, ref_roi, interpolator, spacing, default_value)
        else:
            raise RuntimeError("Unknown transform method for images: {}".format(method))

    def transform_segmentation(self,
                        image: Union[str, vtk.vtkImageData, sitk.Image],
                        ref: Optional[Union[str, sitk.Image, OrientedBox]],
                        method='vtk',
                        spacing=None,
                        default_value=0):
        """
        Variation of transform_image that applies defaults for segmentations.
        This primarily affects the interpolator which should be 'nearest' for segmentations.
        """
        return self.transform_image(image, ref, method,
                                    interpolator='nearest',
                                    spacing=spacing,
                                    default_value=default_value)

    def _transform_image_vtk(self,
                             image: Union[str, vtk.vtkImageData, sitk.Image],
                             roi: OrientedBox,
                             interpolator, spacing, default_value):

        if isinstance(image, str):
            image = sitk.ReadImage(image)
        if isinstance(image, sitk.Image):
            image = sitk_to_vtk_image(image)
        if not isinstance(image, vtk.vtkImageData):
            raise RuntimeError("Invalid image type: {}".format(type(image).__name__))

        out_spacing = native(ndtuple(spacing))
        out_origin = native(roi.origin)
        extent = roi.get_voxel_extent(out_spacing)
        interpolator = get_interpolator(interpolator, 'vtk')

        # extract the coordinate system from the input image and create an affine transformation
        # note that we apply an inverse transform, i.e., from absolute coordinates to image coordinates
        image_roi = OrientedBox()
        image_roi.adjust(image)
        t_in = vtk.vtkTransform()
        t_in.SetMatrix(image_roi.to_matrix(image.GetSpacing()).flatten())
        t_in.Inverse()

        # extract the coordinate system from the reference and create an affine transformation
        # note that we translate from the reference, i.e, its coordinate system, to absolute coordinates
        t_out = vtk.vtkTransform()
        t_out.SetMatrix(roi.to_matrix(out_spacing).flatten())
        t = vtk.vtkGeneralTransform()

        # chain the resulting transforms
        # transformations in their actual order:
        # - transform from the output coordinate system to absolute coordinates
        # - apply the inverse transform (as we map output to input coordinates)
        # - transform the absolute coordinates to the input coordinate system
        t.Concatenate(t_in)
        t.Concatenate(self.inverse.inner)
        t.Concatenate(t_out)

        # ignore the original transform of the image as it is included into the transform
        # for this we create a copy of the image and only pass the pointdata
        image_scalars = image.GetPointData().GetScalars()
        image_dims = image.GetDimensions()
        image = vtk.vtkImageData()
        image.SetDimensions(image_dims)
        image.GetPointData().SetScalars(image_scalars)
        image.Modified()

        # we are using vtkImageReslice to transform the image, however most features will be disabled
        # and any coordinate translation is done via the transform
        reslice = vtk.vtkImageReslice()
        reslice.TransformInputSamplingOff()
        reslice.SetInterpolationMode(interpolator)
        reslice.SetBackgroundLevel(default_value)


        # set the plain input image (without any transform information) and the combined transform
        reslice.SetInputData(image)
        reslice.SetResliceTransform(t)

        # dont let vtk interfere with any
        reslice.SetOutputOrigin(np.zeros(3))
        reslice.SetOutputSpacing(np.ones(3))
        reslice.SetOutputExtent(extent)

        reslice.Update()
        result = reslice.GetOutput()
        result.SetSpacing(out_spacing)
        result.SetOrigin(out_origin)
        result.SetDirectionMatrix(roi.direction.flatten())

        result = vtk_to_sitk_image(result)
        return result

