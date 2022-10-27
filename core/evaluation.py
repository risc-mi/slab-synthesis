#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os
import sys

import SimpleITK as sitk

from libs.metrics import evaluate_overlap_labels, evaluate_overlap_images
from libs.samples import get_sample_info
from libs.util.file import mkdirs, read_json, write_pickle, read_pickle
from run.config import get_default_extension
from libs.util.image import mask_nonempty_slices


def evaluate_samples(sample_root, warps_path, output_path, models,
                     override=True, image_metrics=True, label_metrics=True):

    ext = get_default_extension()
    update_rate = 10
    sample_map, sample_sets, sample_meta = get_sample_info(sample_root, ext=ext)

    split_path = os.path.join(sample_root, 'split.json')
    print("Using split file: {}".format(split_path))
    split = read_json(split_path) if os.path.exists(split_path) else dict()
    split_map = dict((n, s) for s, l in split.items() for n in l)

    mkdirs(output_path)
    score_path = os.path.join(output_path, 'scores.pkl')
    scores = dict()
    if not override and os.path.exists(score_path):
        try:
            scores = read_pickle(score_path)
        except:
            print("Failed to load scores file: {}".format(score_path), file=sys.stderr)
            raise

    def _print_example(_model, _scores):
        _dsc = _scores[_model].get('avg', dict()).get('dsc', None)
        if _dsc is not None:
            print('{}={:.0%}... '.format(_model, _dsc), end='')
        else:
            print('{}=no-labels... '.format(_model, _dsc), end='')

    model_map = dict()
    for model in models:
        model_path = os.path.join(warps_path, model)
        warp_map, _, _ = get_sample_info(model_path, ext=ext)
        model_map[model] = warp_map

    for sample_idx, (sample_id, sample_items) in enumerate(sample_map.items()):
        sample_scores = scores.get(sample_id, dict())
        if override or len(sample_scores.keys()) < (2+len(models)):
            print("Processing {}... ".format(sample_id), end='')
            imgA = sitk.ReadImage(sample_items['A']) if image_metrics else None
            maskA = sitk.ReadImage(sample_items['A_mask'], sitk.sitkUInt8)
            labelA = sitk.ReadImage(sample_items['A_label'], sitk.sitkUInt8) if label_metrics else None
            imgB = sitk.ReadImage(sample_items['B']) if image_metrics else None
            maskB = sitk.ReadImage(sample_items['B_mask'], sitk.sitkUInt8)
            labelB = sitk.ReadImage(sample_items['B_label'], sitk.sitkUInt8) if label_metrics else None
            overlap = mask_nonempty_slices(sitk.And(maskA, maskB))

            if override or 'meta' not in sample_scores:
                sample_scores.setdefault('meta', dict())
                sample_scores['meta'].update(sample_meta[sample_id])
                sample_scores['meta']['split'] = split_map.get(sample_id, 'error')

            """
            calculate metrics for the original slabs
            """
            if override or 'orig' not in sample_scores:
                sample_scores.setdefault('orig', dict())
                if label_metrics:
                    sample_scores['orig'].update(evaluate_overlap_labels(labelA, labelB, overlap))
                if image_metrics:
                    sample_scores['orig'].update(evaluate_overlap_images(imgA, imgB, overlap))
            _print_example('orig', sample_scores)

            """
            calculate metrics for the warped slabs
            """
            for model in models:
                if override or model not in sample_scores:
                    warp_map = model_map[model]
                    warp_result = warp_map.get(sample_id, dict())
                    warped = 0

                    bg = 0
                    if image_metrics:
                        filter = sitk.MinimumMaximumImageFilter()
                        filter.Execute(imgA)
                        bg = filter.GetMinimum()


                    imgA_warped = imgA if image_metrics else None
                    imgB_warped = imgB if image_metrics else None
                    labelA_warped = labelA if label_metrics else None
                    labelB_warped = labelB if label_metrics else None
                    path = warp_result.get('A_warp')
                    if path:
                        dfA = sitk.ReadImage(path, sitk.sitkVectorFloat64)
                        dftA = sitk.DisplacementFieldTransform(dfA)
                        imgA_warped = sitk.Resample(imgA_warped, maskA, dftA, sitk.sitkBSpline, bg) if image_metrics else None
                        labelA_warped = sitk.Resample(labelA_warped, maskA, dftA, sitk.sitkNearestNeighbor) if label_metrics else None
                        warped += 1

                    path = warp_result.get('B_warp')
                    if path:
                        dfB = sitk.ReadImage(path, sitk.sitkVectorFloat64)
                        dftB = sitk.DisplacementFieldTransform(dfB)
                        imgB_warped = sitk.Resample(imgB_warped, maskB, dftB, sitk.sitkBSpline, bg) if image_metrics else None
                        labelB_warped = sitk.Resample(labelB_warped, maskB, dftB, sitk.sitkNearestNeighbor) if label_metrics else None

                    if warped == 1:
                        sample_scores.setdefault(model, dict())
                        if label_metrics:
                            sample_scores[model].update(evaluate_overlap_labels(labelA_warped, labelB_warped, overlap))
                        if image_metrics:
                            sample_scores[model].update(evaluate_overlap_images(imgA_warped, imgB_warped, overlap))
                        _print_example(model, sample_scores)
                    elif warped > 1:
                        print('\nSKIP: Multiple warps defined for sample: {}'.format(sample_id))
                    else:
                        print('\nSKIP: No warp found for sample: {}'.format(sample_id))

            scores[sample_id] = sample_scores

            # save scores every N samples
            if sample_idx % update_rate == 0:
                print("\nsaving scores...")
                write_pickle(scores, score_path)
            print()
    write_pickle(scores, score_path)