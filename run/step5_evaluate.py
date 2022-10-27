#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os

from libs.util.misc import headline
from run.config import get_config, get_result_root
from core.evaluation import evaluate_samples


def main():
    root = get_result_root()
    config = get_config()
    models = config.models
    samples_path = os.path.join(root, 'samples')
    warps_path = os.path.join(root, 'warps')
    output_path = os.path.join(root, 'eval')

    headline('Evaluation: script for calculating evaluation metrics')
    print("The samples path is set to: {}".format(samples_path))
    print("The warp results path is set to: {}".format(warps_path))
    print("The output path is set to: {}".format(output_path))

    evaluate_samples(samples_path, warps_path, output_path,
                     models=models, override=config.override, image_metrics=config.image_metrics, label_metrics=config.label_metrics)


if __name__ == '__main__':
    main()
