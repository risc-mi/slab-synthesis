#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from collections import defaultdict

import SimpleITK as sitk
import numpy as np

from libs.util.misc import nan_clip


def evaluate_overlap_images(imgA, imgB, overlap):
    scores = defaultdict(dict)

    m = sitk.ImageRegistrationMethod()
    m.SetMetricFixedMask(overlap)
    m.SetMetricMovingMask(overlap)
    metrics = {
        'cor': m.SetMetricAsCorrelation,
        'mmi': m.SetMetricAsMattesMutualInformation,
        'msq': m.SetMetricAsMeanSquares,
        #'ncc': lambda: m.SetMetricAsANTSNeighborhoodCorrelation(5)
    }

    imgA = sitk.Cast(imgA, sitk.sitkFloat64)
    imgB = sitk.Cast(imgB, sitk.sitkFloat64)
    for metric, fun in metrics.items():
        fun()
        iv = m.MetricEvaluate(imgA, imgB)
        scores['img'][metric] = iv

    #mask = sitk.GetArrayFromImage(overlap).astype(bool)
    #idiff = sitk.AbsoluteValueDifference(imgA, imgB)
    #imad = np.average(sitk.GetArrayViewFromImage(idiff)[mask])
    #scores['img']['mad'] = imad.item()

    #icorr = corr(imgA, imgB)
    #incc = -np.average(sitk.GetArrayViewFromImage(icorr)[mask])
    #scores['img']['ncc'] = incc.item()

    return scores


def evaluate_overlap_labels(labelA, labelB, mask):
    scores = dict()

    labelA = sitk.MaskNegated(labelA, mask, 0, 1)
    labelB = sitk.MaskNegated(labelB, mask, 0, 1)

    overlap = sitk.LabelOverlapMeasuresImageFilter()
    overlap.Execute(labelA, labelB)

    labels = set(np.unique([sitk.GetArrayViewFromImage(labelA), sitk.GetArrayViewFromImage(labelB)]))
    labels.remove(0)
    for label in labels:
        label = int(label)
        label_score = dict()
        label_score['vs'] = overlap.GetVolumeSimilarity(label)
        label_score['dsc'] = overlap.GetDiceCoefficient(label)
        label_score['jc'] = overlap.GetJaccardCoefficient(label)
        label_score['mo'] = overlap.GetMeanOverlap(label)
        label_score['uo'] = overlap.GetUnionOverlap(label)
        label_score['fpe'] = nan_clip(overlap.GetFalsePositiveError(label), 0.0, 1.0)
        label_score['fne'] = nan_clip(overlap.GetFalseNegativeError(label), 0.0, 1.0)


        segA = labelA == label
        segB = labelB == label

        label_ahd = np.nan
        try:
            stats = sitk.HausdorffDistanceImageFilter()
            stats.Execute(segA, segB)
            label_ahd = stats.GetAverageHausdorffDistance()
        except:
            pass
        label_score['ahd'] = label_ahd

        scores['label{}'.format(label)] = label_score


    metrics = sorted(set(m for l, ls in scores.items() for m in ls.keys()))
    avgs = dict((m, np.nanmean(list(ls[m] for ls in scores.values()))) for m in metrics)
    result = dict()
    result['avg'] = avgs
    result.update(scores)
    return result