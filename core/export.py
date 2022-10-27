#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os
import time
import traceback
import warnings
from collections import defaultdict

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use('tkagg')

import numpy as np

from libs.util.file import mkdirs, read_json, read_pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def _load_scores(root):
    score_path = os.path.join(root, 'scores.pkl')
    if not os.path.exists(score_path):
        raise RuntimeError("No score file to evaluate at: {}".format(score_path))
    scores = None
    for i in range(10):
        try:
            scores = read_pickle(score_path)
            break
        except:
            time.sleep(0.10)
    if scores is None:
        traceback.print_exc()
        raise RuntimeError("Unable to load scores from: {}".format(score_path))
    return scores


def _get_metric_args(metric):
    default = {
        'sign': 1
    }

    custom = {
        'ahd': {
            'limits': (0, 5),
            'alias': "Average Hausdorff (AHD)"
        },
        'dsc': {
            'limits': (0, 1),
            'alias': "Dice Similarity Coefficient (DSC)"
        },
        'fne': {
            'limits':  (0, 1),
            'alias': "False Negative Error (FNE)"
        },
        'fpe': {
            'limits': (0, 1),
            'alias': "False Positive Error (FPE)"
        },
        'vs': {
            'limits': (-1, 1),
            'alias': "Volume Similarity (VS)"
        },
        'jc': {
            'limits':  (0, 1),
            'alias': "Jaccard Similarity Coefficient (JSC)"
        },
        'mo': {
            'limits': (0, 1),
            'alias': "Mean Overlap (MO)"
        },
        'uo': {
            'limits': (0, 1),
            'alias': "Union Overlap (UO)"
        },
        'cor': {
            'limits': (0, 1),
            'alias': "Normalized Cross Correlation (NCC)",
            'sign': -1,
        },
        'mmi': {
            'limits': None,
            'alias': "Mattes Mutual Information (MMI)",
            'sign': -1,
        },
        'msq': {
            'limits': (0, 1e6),
            'alias': "Mean Squared Errors (MSE)",
            'log': True
        },
    }
    args = default.copy()
    if metric in custom:
        args.update(custom[metric])
    return args


def _metric_table(scores, split, evals, set_group_to_samples, set_to_samples, export_metric, output_path):

    target_sets = set(set_to_samples.keys())
    if len(target_sets) == 2:
        target_sets.remove('full')

    for split_name in split.keys():
        metric_name = export_metric
        target_evals = evals

        metric_args = _get_metric_args(metric_name)
        metric_sign = metric_args.get('sign', 1.0)

        samples = set(split[split_name])
        sample_scores = defaultdict(dict)
        for sample_id, eval_keys in scores.items():
            item_id = sample_id.split('-')[1]
            if item_id in samples:
                for eval_key in eval_keys:
                    if eval_key in target_evals:
                        score = scores[sample_id][eval_key]
                        score_value = score.get('avg', dict()).get(metric_name)
                        if score_value is None:
                            score_value = score.get('img', dict()).get(metric_name)
                        sample_scores[sample_id][eval_key] = score_value

        results = defaultdict(dict)
        result_keys = None

        set_groups = sorted(set_group_to_samples.keys())
        for set_key, group_name in set_groups:
            if set_key in target_sets:
                set_name = set_key
                samples = set(sample_id
                              for sample_id in set_group_to_samples[(set_key, group_name)]
                              if sample_id.split('-')[1] in split[split_name])
                if len(samples) > 0:
                    group_values = list((sample_id, sample_scores[sample_id])
                                        for sample_id in set(sample_scores.keys()).intersection(samples))
                    set_group = '{}-{}'.format(set_name, group_name)

                    for eval_key in target_evals:
                        eval_name = eval_key
                        eval_values = list(v.get(eval_key, np.nan) for _, v in group_values)
                        eval_values = np.multiply(eval_values, metric_sign)
                        eval_result = {
                            'mean': np.mean(eval_values),
                            'std': np.std(eval_values),
                            'median': np.median(eval_values),
                            'min': np.min(eval_values),
                            'max': np.max(eval_values),
                            'q1': np.quantile(eval_values, 0.25),
                            'q3': np.quantile(eval_values, 0.75)
                        }
                        if result_keys is None:
                            result_keys = sorted(eval_result.keys())
                        results[set_group][eval_name] = eval_result

        result_path = os.path.join(output_path, 'eval_{}.xlsx'.format(split_name))
        writer = pd.ExcelWriter(result_path, engine='xlsxwriter')
        for result_key in result_keys:
            result_table = defaultdict(dict)
            for set_group, set_results in results.items():
                for eval_name, eval_results in set_results.items():
                    result_table[set_group][eval_name] = eval_results[result_key]
            df = pd.DataFrame.from_dict(result_table, orient='index')
            df.to_excel(writer, sheet_name=result_key)
        writer.save()


def _metric_set_plots(scores, metrics, evals,
                      split, split_map,
                      set_to_samples,
                      output_path):

    target_sets = set(set_to_samples.keys())
    target_splits = {'train', 'test'}

    if len(target_sets) == 2:
        target_sets.remove('full')

    splits = sorted(split.keys()) + ['all']
    splits = list(s for s in splits if s in target_splits)

    for split_name in splits:
        print("-- processing split {}...".format(split_name))
        sub_dir = os.path.join(output_path, split_name)
        mkdirs(sub_dir)
        exporter = PlotHelper(sub_dir)
        exporter.setup()
        for metric_name in sorted(metrics):
            print("plotting metric {}...".format(metric_name))
            metric_set_stats = list()
            for eval_key in evals:
                for set_key, samples in set_to_samples.items():
                    set_name = set_key
                    if set_key in target_sets:
                        for sample_id in samples:
                            item_id = sample_id.split('-')[1]
                            if split_name == 'all' or split_map.get(item_id, 'test') == split_name:
                                if eval_key in scores[sample_id]:
                                    eval_name = eval_key
                                    score = scores[sample_id][eval_key]
                                    score_value = score.get('avg', dict()).get(metric_name)
                                    if score_value is None:
                                        score_value = score.get('img', dict()).get(metric_name)
                                    if score_value is not None:
                                        entry = dict()
                                        entry['val'] = score_value
                                        entry['eval'] = eval_name
                                        entry['set'] = set_name
                                        metric_set_stats.append(entry)

            metric_set_stats = pd.DataFrame.from_records(metric_set_stats)

            order = sorted(target_sets)
            hue_order = sorted(evals)
            exporter.metric_plot_hue(metric_name, metric_set_stats,
                                     hue='eval', cat_name='set', order=order, hue_order=hue_order)


class PlotHelper:
    def __init__(self, output_path):
        self.output_path = output_path
        self.plot_eps = False
        self.plot_legend = True

    def setup(self):
        plt.rcParams.update({'font.size': 12})
        plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('axes', labelsize=14)

    def metric_plot_hue(self, metric, data, cat_name=None, hue=None, order=None, hue_order=None):
        metric_config = _get_metric_args(metric)
        metric_alias = metric_config.get('alias', metric)
        metric_limits = metric_config.get('limits')
        metric_sign = metric_config.get('sign', 1.0)

        plot_title = metric_alias
        plot_name = "{}".format(metric)

        data['val'] = data['val'].astype(np.float64) * metric_sign

        plt.figure(1)
        ax = sns.boxplot(data=data, x=cat_name, y='val',
                         fliersize=0, palette='colorblind',
                         hue=hue, order=order, hue_order=hue_order,
                         linewidth=1)
        ax.set_title(plot_title)
        ax.set_ylabel('')
        ax.set_xlabel('')

        linear = not metric_config.get('log', False)
        ax.set_yscale('linear' if linear else 'log')
        if metric_limits is not None:
            lims = metric_limits if linear else np.clip(metric_limits, a_min=0.1, a_max=None)
            ax.set_ylim(*tuple(lims))
        if not self.plot_legend:
            plt.legend().remove()
        else:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                      fancybox=True, shadow=True, ncol=5)
        plt.gcf().set_dpi(300)
        if self.plot_eps:
            plt_path = os.path.join(self.output_path, "{}.eps".format(plot_name))
            plt.savefig(plt_path, format='eps')
        else:
            plt_path = os.path.join(self.output_path, "{}.png".format(plot_name))
            plt.savefig(plt_path)
        plt.clf()


def export_results(input_root, output_path, split_path, export_metric, export_plots):
    mkdirs(output_path)
    scores = _load_scores(input_root)

    split = read_json(split_path) if os.path.exists(split_path) else dict()
    if len(split) == 0:
        raise RuntimeError("No splits available!")
    split_map = dict((n, s) for s, l in split.items() for n in l)

    set_to_samples = defaultdict(set)
    set_group_to_samples = defaultdict(set)
    set_to_groups = defaultdict(set)
    metrics = set()
    models = set()
    evals = set()
    for sample_id, sample_info in scores.items():
        sample_meta = sample_info['meta']
        sample_set = sample_meta['set']
        sample_group = sample_meta['group']
        set_to_samples[sample_set].add(sample_id)
        set_to_samples['full'].add(sample_id)
        set_group_to_samples[(sample_set, sample_group)].add(sample_id)
        set_group_to_samples[(sample_set, 'any')].add(sample_id)
        set_group_to_samples[('full', sample_group)].add(sample_id)
        set_group_to_samples[('full', 'any')].add(sample_id)
        set_to_groups[sample_set].add(sample_group)
        set_to_groups['full'].add(sample_group)

        sample_keys = set(sample_info.keys())
        metric_evals = sample_keys.difference({'meta'})
        for eval_key in metric_evals:
            eval_groups = sample_info[eval_key]
            for eval_group, eval_stats in eval_groups.items():
                metrics.update(eval_stats.keys())

        model_evals = metric_evals.difference({'orig'})
        models.update(model_evals)
        evals.update(metric_evals)

    if export_plots:
        _metric_set_plots(scores, metrics, evals,
                          split, split_map, set_to_samples,
                          output_path)

    _metric_table(scores, split, evals, set_group_to_samples, set_to_samples, export_metric, output_path)