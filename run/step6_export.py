#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os
import warnings

from run.config import get_config, get_result_root
from core.export import export_results

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use('tkagg')

from libs.util.misc import headline


def main():
    root = get_result_root()
    config = get_config()
    split_path = os.path.join(root, 'pp', 'split.json')
    input_path = os.path.join(root, 'eval')
    output_path = os.path.join(root, 'export')

    headline('Evaluation: script for creating evaluation plots and tables')
    print("The eval path is set to: {}".format(input_path))
    print("The output path is set to: {}".format(output_path))
    print("Using split file: {}".format(split_path))

    export_results(input_path, output_path, split_path, config.export_metric,  config.export_plots)


if __name__ == '__main__':
    main()
