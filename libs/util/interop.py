#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os
import sys
import traceback
from typing import Union, Optional

import SimpleITK as sitk
import numpy as np
import vtk

from libs.util.file import SafeTemporaryDirectory
from libs.util.misc import native, ndtuple
from libs.util.oriented_box import OrientedBox


def sitk_to_vtk_image(img: sitk.Image,
                      method='arr') -> vtk.vtkImageData:
    """
    converts a simpleitk image to a vtk image
    :param img: simpleitk image to convert
    :param method: method to use for conversion of the pixel data, either 'file' or 'arr'
    :return: the resulting vtkImageData
    Note: method 'file' may be more complete and robust as it leaves the actual conversion to
    the vtk/simpleitk IO classes. 'arr' on the other hand should be faster.
    """
    if method == 'file':
        result = _sitk_to_vtk_image_file(img)
    elif method == 'arr':
        result = _sitk_to_vtk_image_arr(img)
    else:
        raise RuntimeError("Unknown conversion method: {}".format(method))

    result.SetOrigin(img.GetOrigin())
    result.SetDirectionMatrix(img.GetDirection())
    result.SetSpacing(img.GetSpacing())

    return result


def _sitk_to_vtk_image_file(img: sitk.Image) -> vtk.vtkImageData:
    """
    converts a simpleitk image to a vtk image using file IO
    """
    with SafeTemporaryDirectory() as tmp:
        tmp_file = os.path.join(tmp, 'tmp.mha')
        sitk.WriteImage(img, tmp_file, False)

        reader = vtk.vtkMetaImageReader()
        reader.SetFileName(tmp_file)
        reader.Update()
        return reader.GetOutput()


def _sitk_to_vtk_image_arr(img: sitk.Image) -> vtk.vtkImageData:
    """
    converts a simpleitk image to a vtk image by converting the buffer data
    """
    arr = sitk.GetArrayFromImage(img)

    from vtk.util import numpy_support as vns
    varr = vns.numpy_to_vtk(arr.ravel(), deep=True)
    varr.SetNumberOfComponents(img.GetNumberOfComponentsPerPixel())
    result = vtk.vtkImageData()
    result.SetDimensions(*img.GetSize())
    result.GetPointData().SetScalars(varr)
    result.Modified()

    return result


def vtk_to_sitk_image(img: vtk.vtkImageData,
                      method='arr',
                      outputPixelType=sitk.sitkUnknown) -> sitk.Image:
    """
    converts a vtk image to a simpleitk image
    :param img: vtk image data to convert
    :param method: method to use for conversion of the pixel data, either 'file' or 'arr'
    :param outputPixelType: (optional) pixel type for the simpleitk image
    :return: the resulting simpleitk image
    Note: method 'file' may be more complete and robust as it leaves the actual conversion to
    the vtk/simpleitk IO classes. 'arr' on the other hand should be faster.
    """
    if method == 'file':
        result = _vtk_to_sitk_image_file(img, outputPixelType)
    elif method == 'arr':
        result = _vtk_to_sitk_image_arr(img, outputPixelType)
    else:
        raise RuntimeError("Unknown conversion method: {}".format(method))

    result.SetOrigin(img.GetOrigin())
    result.SetDirection(vtk_to_numpy_matrix(img.GetDirectionMatrix()).flatten())
    result.SetSpacing(img.GetSpacing())

    return result


def _vtk_to_sitk_image_file(img: vtk.vtkImageData, outputPixelType) -> sitk.Image:
    """
    converts a vtk image to a simpleitk image using file IO
    """
    with SafeTemporaryDirectory() as tmp:
        tmp_file = os.path.join(tmp, 'tmp.mha')
        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName(tmp_file)
        writer.SetInputData(img)
        writer.Write()
        return sitk.ReadImage(tmp_file, outputPixelType)


def _vtk_to_sitk_image_arr(img: vtk.vtkImageData, outputPixelType) -> sitk.Image:
    """
    converts a vtk image to a simpleitk image using file IO
    """
    isVector = img.GetNumberOfScalarComponents() > 1
    arr = vtk_to_numpy_image(img)
    simg = sitk.GetImageFromArray(arr, isVector)
    if outputPixelType != sitk.sitkUnknown:
        simg = sitk.Cast(simg, outputPixelType)

    mat = vtk_to_numpy_matrix(img.GetDirectionMatrix())
    simg.SetOrigin(img.GetOrigin())
    simg.SetDirection(mat.flatten())
    simg.SetSpacing(img.GetSpacing())

    return simg


def vtk_to_numpy_matrix(mat: Union[vtk.vtkMatrix3x3, vtk.vtkMatrix4x4]):
    if isinstance(mat, vtk.vtkMatrix3x3):
        shape = (3, 3)
    elif isinstance(mat, vtk.vtkMatrix4x4):
        shape = (4, 4)
    else:
        raise RuntimeError("No supported vtk matrix type: ".format(type(mat).__name__))

    res = np.zeros(shape)
    for i, j in np.ndindex(shape):
        res[i, j] = mat.GetElement(i, j)
    return res


def vtk_to_numpy_image(img: vtk.vtkImageData) -> np.ndarray:
    """
    converts a vtk image to a numpy array.
    note that for an image of w x h x d voxels and c channels, the
    array is structured as d x h x w x c
    :param img: vtk image
    :return: corresponding numpy array
    """
    from vtk.util import numpy_support as vns
    varr = img.GetPointData().GetScalars()
    shape = tuple(reversed(img.GetDimensions()))
    isVector = img.GetNumberOfScalarComponents() > 1
    if isVector:
        shape = shape + (img.GetNumberOfScalarComponents(), )
    arr = vns.vtk_to_numpy(varr).reshape(shape)
    return arr


def get_interpolator(name: str, target='sitk'):
    """
    translates an interpolator name to its representation in the target library
    :param name: interpolator name, e.g., 'nearest', 'linear', 'spline', returns default for None
    :param target: either 'sitk' or 'vtk'
    :return:
    """
    if target == 'sitk':
        return get_interpolator_sitk(name)
    elif target == 'vtk':
        return get_interpolator_vtk(name)
    else:
        raise RuntimeError("No valid target for get_interpolator: {}".format(target))


def get_interpolator_sitk(name: str):
    if name is None:
        return sitk.sitkLinear
    map = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'spline': sitk.sitkBSpline,
    }
    res = map.get(name)
    if res is None:
        raise RuntimeError("No SimpleITK interpolator for '{}'".format(name))
    return res


def get_interpolator_vtk(name: str):
    if name is None:
        return vtk.VTK_LINEAR_INTERPOLATION
    map = {
        'nearest': vtk.VTK_NEAREST_INTERPOLATION,
        'linear': vtk.VTK_LINEAR_INTERPOLATION,
        'spline': vtk.VTK_CUBIC_INTERPOLATION,
    }
    res = map.get(name)
    if res is None:
        raise RuntimeError("No VTK interpolator for '{}'".format(name))
    return res


def sitk_to_vtk_transform(t: sitk.Transform, roi: Optional[OrientedBox] = None, spacing=1.0, simplify=True):
    """
    converts a SimpleITK transform to a vtk transform object
    :param t: simpleitk transform to convert
    :param roi: target ROI, relevant for nonlinear transforms
    :param spacing: spacing of the target ROI, relevant for nonlinear transforms, specifically displacement fields
    :param simplify: whether to simplify the transform before conversion by flattening and combining affine transforms
    :return: resulting vtk transform
    """
    return SimpleITKtoVTK.convert(t=t, roi=roi, spacing=spacing, simplify=simplify)


def vtk_to_sitk_transform(t: vtk.vtkAbstractTransform, roi: Optional[OrientedBox] = None, spacing=1.0, simplify=True):
    """
    converts a VTK transform to a simpleitk transform object
    :param t: vtk transform to convert
    :param roi: target ROI, relevant for nonlinear transforms
    :param spacing: spacing of the target ROI, relevant for nonlinear transforms, specifically displacement fields
    :param simplify: whether to simplify the resulting transform by flattening and combining affine transforms
    :return: resulting simpleitk transform
    """
    return VTKtoSimpleITK.convert(t=t, roi=roi, spacing=spacing, simplify=simplify)


def numpy_to_vtk_matrix(mat: np.ndarray):
    if mat.shape == (3, 3):
        vmat = vtk.vtkMatrix3x3()
    elif mat.shape == (4, 4):
        vmat = vtk.vtkMatrix4x4()
    else:
        raise RuntimeError("Unable to convert numpy array of shape {} to vtk matrix.".format('x'.join(str(s) for s in mat.shape)))
    vmat.DeepCopy(mat.flatten())
    return vmat


def sitk_to_displacement(t: sitk.Transform, roi: OrientedBox, spacing=1.0) -> sitk.Image:
    """
    Converts a generic SimpleITK transform to a displacement field within the given region of interest
    :param t: transform to convert
    :param roi: region of interest that defines the boundaries of the displacement field
    :param spacing: voxel spacing of the displacement field
    :return: resulting displacement field
    """
    size = native(roi.get_voxel_size(spacing))
    filter = sitk.TransformToDisplacementFieldFilter()
    filter.SetOutputOrigin(native(roi.origin))
    filter.SetOutputSpacing(native(ndtuple(spacing)))
    filter.SetOutputDirection(native(roi.direction.flatten()))
    filter.SetSize(size)
    return filter.Execute(t)


class SimpleITKtoVTK:
    """
    Converts SimpleITK transform types to its VTK counterparts.
    This class is loosely based on https://github.com/Slicer/Slicer/blob/master/Libs/MRML/Core/vtkITKTransformConverter.h
    """
    def __init__(self, roi: Optional[OrientedBox]=None, spacing=1.0):
        self.roi = roi
        self.spacing = spacing

    @staticmethod
    def convert(t: sitk.Transform, roi: Optional[OrientedBox] = None, spacing=1.0, simplify=True):
        if simplify:
            t = simplify_transform(t)
        return SimpleITKtoVTK(roi=roi, spacing=spacing)._convert(t)

    def _convert(self, t: sitk.Transform):
        """
        Tries to convert the given SimpleITK Transform (and possibly its composite elements) to its respective
        VTK counterpart. In case the transform cannot be converted directly, the SimpleITK transform will first be
        converted to a displacement field using the specified region of interest (if supplied, otherwise it fails).
        :param t: SimpleITK transform
        :return: the resulting VTK transform
        """
        if t.GetTransformEnum() == sitk.sitkIdentity:
            return vtk.vtkIdentityTransform()
        if t.GetTransformEnum() == sitk.sitkComposite:
            return self._convert_composite(t)
        if t.GetTransformEnum() == sitk.sitkBSplineTransform:
            return self._convert_bspline(t)
        if t.GetTransformEnum() == sitk.sitkTranslation:
            return self._convert_translation(t)
        if t.GetTransformEnum() == sitk.sitkScale:
            return self._convert_scale(t)
        if t.GetTransformEnum() == sitk.sitkAffine:
            return self._convert_affine(t)
        if t.GetTransformEnum() == sitk.sitkDisplacementField:
            return self._convert_displacement(t)
        # add further transform types here as needed

        msg = "No conversion implemented for transform of type: {}".format(t.GetName())
        if self.roi is None:
            raise RuntimeError(msg)

        print(msg, file=sys.stderr)
        print("Falling back to a generic displacement field transform."
              "This might be much less efficient than the original transform.", file=sys.stderr)
        return self._fallback_displacement(t)

    def _fallback_displacement(self, t):
        """
        First converts the transform to a displacement field (an operation that should work on any SimpleITK transform)
        and then to VTK. This method requires the ROI and spacing to be specified.
        """
        assert(self.roi is not None)
        if not self.roi.valid:
            raise RuntimeError("Unable to create a displacement field as the supplied RoI is not valid!")

        sroi = OrientedBox()
        sroi.adjust(self.roi)
        sroi.standardize()
        df = sitk_to_displacement(t, sroi, self.spacing)

        vtk_dt = sitk_to_vtk_image(df)
        vtk_t = vtk.vtkGridTransform()
        vtk_t.SetDisplacementGridData(vtk_dt)
        vtk_t.Update()
        return vtk_t

    def _convert_composite(self, t: sitk.Transform):
        t = sitk.CompositeTransform(t)
        result = vtk.vtkGeneralTransform()
        for idx in range(t.GetNumberOfTransforms()):
            sub = t.GetNthTransform(idx)
            try:
                vtk_t = self._convert(sub)
                result.Concatenate(vtk_t)
            except Exception as ex:
                traceback.print_exc()
                raise RuntimeError("Failed to convert transform element: {} ".format(ex))
        return result

    def _convert_identity(self, t: sitk.Transform):
        return vtk.vtkIdentityTransform()

    def _convert_translation(self, t: sitk.Transform):
        t = sitk.TranslationTransform(t)
        offset = np.asarray(t.GetOffset())
        vtk_t = vtk.vtkTransform()
        vtk_t.Translate(offset)
        return vtk_t

    def _convert_scale(self, t: sitk.Transform):
        t = sitk.ScaleTransform(t)
        scale = t.GetScale()
        center = np.asarray(t.GetCenter())
        vtk_t = vtk.vtkTransform()
        vtk_t.Translate(-center)
        vtk_t.Scale(scale)
        vtk_t.Translate(center)
        return vtk_t

    def _convert_displacement(self, t: sitk.Transform):
        t = sitk.DisplacementFieldTransform(t)
        df = t.GetDisplacementField()
        vtk_dt = sitk_to_vtk_image(df)
        vtk_t = vtk.vtkGridTransform()
        vtk_t.SetDisplacementGridData(vtk_dt)
        vtk_t.Update()
        return vtk_t

    def _convert_affine(self, t: sitk.Transform):
        t = sitk.AffineTransform(t)
        if all(hasattr(t, attr) for attr in ('GetMatrix', 'GetTranslation')):
            vtkMat = np.eye(4)
            itkMat = np.reshape(t.GetMatrix(), (3, 3))
            offset = np.asarray(t.GetTranslation())
            vtkMat[:3, :3] = itkMat
            vtkMat[:3, 3] = offset

            itkCenter = (0, 0, 0)
            if hasattr(t, 'GetCenter'):
                itkCenter = np.asarray(t.GetCenter())
            if np.linalg.norm(itkCenter) > 0:
                transMat = np.eye(4)
                transMat[:3, 3] = -itkCenter
                np.matmul(vtkMat, transMat, out=vtkMat)
                transMat[:3, 3] = itkCenter
                np.matmul(transMat, vtkMat, out=vtkMat)
        else:
            raise RuntimeError("Unsupported SimpleITK Transform type: {}".format(t.GetClassName()))

        vtk_t = vtk.vtkTransform()
        vtk_t.SetMatrix(vtkMat.flatten())
        return vtk_t

    def _convert_bspline(self, t: sitk.Transform) -> vtk.vtkAbstractTransform:
        """
        This method converts a bspline transform from SimpleITK to VTK.
        Note that, SimpleITK bsplines are oriented, while VTK bsplines are not:
        To compensate for the missing funcionality, we have to simulate the orientation by chaining
        additional transforms before and after the application of the bspline transform.
        Also, the actual bspline coefficient vectors have to undergo the same orientation transform.
        Outline of the steps:
        - extract orientation matrix
        - transform coefficient vectors by matrix
        - transform origin by matrix
        - create orient-transform from matrix
        - chain vtk transforms: orient > bspline > inverse orient
        """
        # build a coefficient vector image
        t = sitk.BSplineTransform(t)
        cimg = sitk.Compose(t.GetCoefficientImages())

        # get orientation matrix
        mat3 = np.asarray(cimg.GetDirection()).reshape((3, 3))

        # if any other than identity was used we need to reorient
        reorient = np.any(mat3 != np.eye(3))
        if reorient:
            # transform the coefficients to the orientation and build a new image
            carr = sitk.GetArrayFromImage(cimg)
            t_carr = carr[..., :].dot(mat3.T)
            t_cimg = sitk.GetImageFromArray(t_carr, isVector=True)
            t_cimg.CopyInformation(cimg)

            # prepare the orientation matrix for vtk
            mat4 = np.eye(4)
            mat4[:3, :3] = mat3
            vmat = numpy_to_vtk_matrix(mat4)

            # create a transform to apply the orientation matrix
            orient = vtk.vtkMatrixToLinearTransform()
            orient.SetInput(vmat)
            orient.Update()

            # also convert the origin to the new orientation
            t_origin = orient.TransformPoint(t_cimg.GetOrigin())
        else:
            # identity orientation: nothing to do
            t_cimg = cimg
            t_origin = cimg.GetOrigin()

        # convert the transformed coefficient image and apply the new orientation
        vtk_img = sitk_to_vtk_image(t_cimg)
        vtk_img.SetOrigin(t_origin)

        # having the direction matrix set apparently screws up everything so we need to unset it here
        vtk_img.SetDirectionMatrix(numpy_to_vtk_matrix(np.eye(3)))

        # create the actual bspline transform
        spline = vtk.vtkBSplineTransform()
        spline.SetDebug(True)
        spline.SetCoefficientData(vtk_img)
        spline.SetBorderModeToZero()
        spline.Update()

        if reorient:
            # chain the transforms: orient > bspline > reverse-orient
            vtk_t = vtk.vtkGeneralTransform()
            vtk_t.Concatenate(orient)
            vtk_t.Concatenate(spline)
            vtk_t.Concatenate(orient.GetInverse())
            return vtk_t
        else:
            # return only the b-spline transform as no orienting is needed
            return spline


class VTKtoSimpleITK:
    """
    Converts VTK transform types to its SimpleITK counterparts.
    """

    def __init__(self, roi: Optional[OrientedBox]=None, spacing=1.0):
        self.roi = roi
        self.spacing = spacing

    @staticmethod
    def convert(t: vtk.vtkAbstractTransform, roi: Optional[OrientedBox] = None, spacing=1.0, simplify=True):
        sitk_t = VTKtoSimpleITK(roi=roi, spacing=spacing)._convert(t)
        if simplify:
            sitk_t = simplify_transform(sitk_t)
        return sitk_t

    def _convert(self, t: vtk.vtkAbstractTransform, inverse=False):
        """
        Tries to convert the given SimpleITK Transform (and possibly its composite elements) to its respective
        VTK counterpart. In case the transform cannot be converted directly, the SimpleITK transform will first be
        converted to a displacement field using the specified region of interest (if supplied, otherwise it fails).
        :param t: VTK transform
        :return: the resulting VTK transform
        """
        try:
            if isinstance(t, vtk.vtkGeneralTransform):
                return self._convert_general(t, inverse)
            if isinstance(t, vtk.vtkHomogeneousTransform):
                return self._convert_homogenous(t, inverse)
            if isinstance(t, vtk.vtkBSplineTransform):
                return self._convert_bspline(t, inverse)
        except Exception as ex:
            print("A direct conversion failed: {}".format(ex))

        # add further transform types here as needed

        msg = "Failed to convert transform of type: {}".format(t.GetClassName())
        if self.roi is None:
            raise RuntimeError(msg)

        print(msg, file=sys.stderr)
        print("Falling back to a generic displacement field transform."
              "This might be much less efficient than the original transform.", file=sys.stderr)
        return self._fallback_displacement(t)

    def _convert_general(self, t: vtk.vtkGeneralTransform, inverse):
        result = sitk.CompositeTransform(3)
        if t.GetInverseFlag():
            t = t.GetInverse()
            inverse = not inverse
        indices = range(t.GetNumberOfConcatenatedTransforms())
        indices = reversed(indices) if not inverse else indices
        for idx in indices:
            sub = t.GetConcatenatedTransform(idx)
            sub = self._convert(sub, inverse)
            result.AddTransform(sub)
        return result

    def _convert_homogenous(self, t: vtk.vtkHomogeneousTransform, inverse):
        result = sitk.AffineTransform(3)
        if inverse:
            t = t.GetInverse()
        mat4 = vtk_to_numpy_matrix(t.GetMatrix())
        mat3 = mat4[:3, :3]
        offset = mat4[:3, 3]
        result.SetTranslation(native(offset))
        result.SetMatrix(native(mat3.flatten()))
        return result

    def _convert_bspline(self, t: vtk.vtkBSplineTransform, inverse):
        if t.GetInverseFlag():
            raise RuntimeError("Inverse BSplines are not supported in SimpleITK")
        if t.GetDisplacementScale() != 1.0:
            raise RuntimeError("Scaled BSplines are not supported in SimpleITK")
        if t.GetBorderMode() != vtk.VTK_BSPLINE_ZERO_AT_BORDER:
            print("WARNING: SimpleITK BSplines only support the border mode 'ZeroAtBorder', "
                  "the encountered mode '{}' will be ignored.".format(t.GetBorderModeAsString()), file=sys.stderr)
        vtk_img = t.GetCoefficientData()
        img = vtk_to_sitk_image(vtk_img)
        img_list = list(sitk.VectorIndexSelectionCast(img, idx)
                        for idx in range(img.GetNumberOfComponentsPerPixel()))
        result = sitk.BSplineTransform(img_list)
        return result

    def _fallback_displacement(self, t: vtk.vtkAbstractTransform):
        """
        First converts the transform to a displacement field (an operation that should work on any SimpleITK transform)
        and then to SimpleITK. This method requires the ROI and spacing to be specified.
        """
        assert(self.roi is not None)
        if not self.roi.valid:
            raise RuntimeError("Unable to create a displacement field as the supplied RoI is not valid!")

        if t.GetInverseFlag():
            raise RuntimeError("Inverse displacement fields are not supported in SimpleITK")

        vtk_df = vtk_to_displacement(t, self.roi, self.spacing)
        df = vtk_to_sitk_image(vtk_df)
        dt = sitk.DisplacementFieldTransform(df)
        return dt


def vtk_to_displacement(t: vtk.vtkAbstractTransform, roi: OrientedBox, spacing=1.0) -> vtk.vtkImageData:
    """
    Converts a VTK transform to a VTK displacement image within the given region of interest
    Note: the roi will be standardized, as VTK does not support custom orientations for displacement fields.
    :param t: transform to convert
    :param roi: region of interest that defines the boundaries of the displacement field
    :param spacing: voxel spacing of the displacement field
    :return: resulting displacement field
    """
    sroi = OrientedBox()
    sroi.adjust(roi)
    sroi.standardize()

    spacing = native(ndtuple(spacing))
    origin = native(sroi.origin)
    extent = native(sroi.get_voxel_extent(spacing))

    filter = vtk.vtkTransformToGrid()
    filter.SetInput(t)
    filter.SetGridOrigin(origin)
    filter.SetGridSpacing(spacing)
    filter.SetGridExtent(extent)
    filter.Update()
    return filter.GetOutput()


def simplify_transform(t: sitk.Transform):
    """
    simplifies a transform by flattening the internal hierarchy and combining affine transforms
    """
    if t.GetTransformEnum() == sitk.sitkComposite:
        t = sitk.CompositeTransform(t)
        t.FlattenTransform()
        t = _compress_transform(t)
        if t.GetNumberOfTransforms() == 1:
            t = t.GetBackTransform().Downcast()
    return t


def transform_to_list(t: sitk.CompositeTransform):
    """
    for a given transform, returns the list of (recursively flattened) child transforms (or itself if it is no composite)
    """
    if t.GetTransformEnum() == sitk.sitkComposite:
        t = sitk.CompositeTransform(t)
        return list(e.Downcast()
                    for i in range(t.GetNumberOfTransforms())
                    for e in transform_to_list(t.GetNthTransform(i)))
    return [t]


def _compress_transform(t: sitk.CompositeTransform):
    d = t.GetDimension()
    l1 = transform_to_list(t)
    mat = None
    t.ClearTransforms()
    for e1 in l1+[None]:
        # try to extract an affine transform
        emat = None
        if hasattr(e1, 'GetMatrix'):
            emat = np.eye(d+1)
            emat[:d, :d] = np.asanyarray(e1.GetMatrix()).reshape([d]*2)
            if hasattr(e1, 'GetCenter'):
                cmat = np.eye(d+1)
                cmat[:d, d] = np.negative(e1.GetCenter())
                emat = np.matmul(emat, cmat)
                cmat[:d, d] *= -1
                emat = np.matmul(cmat, emat)

        if hasattr(e1, 'GetTranslation'):
            tmat = np.eye(d+1)
            tmat[:d, d] = e1.GetTranslation()
            emat = np.matmul(tmat, emat) if emat is not None else emat

        # was that successfull?
        if emat is not None:
            # concatenate the affine transform
            mat = np.matmul(mat, emat) if mat is not None else emat
        else:
            # add the previous concatenated transforms
            if mat is not None:
                e2 = sitk.AffineTransform(t.GetDimension())
                e2.SetMatrix(native(mat[:d, :d].flatten()))
                e2.SetTranslation(native(mat[:d, d]))
                t.AddTransform(e2)
                mat = None
            # add the current non-affine transform
            if e1 is not None:
                t.AddTransform(e1)
    return t