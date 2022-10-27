#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os

from libs.pipeline import get_settings
from libs.util.misc import headline
from run.config import get_config, get_result_root, get_resource_root
from core.preprocess import preprocess_all


def main():
    config = get_config()
    skip = config.skip
    override = config.override
    input_root = get_resource_root()
    result_root = get_result_root()
    output_path = os.path.join(result_root, "pp")

    headline("Decathlon Dataset Preprocessing")
    print("The input folder is set to: {}".format(input_root))
    print("The output folder is set to: {}".format(output_path))
    print("Skip tasks: {}".format(', '.join(str(id) for id in sorted(skip)) if skip else 'None'))
    print()

    settings = get_settings(config.spacing)
    preprocess_all(input_root, output_path, settings, override, skip)


if __name__ == '__main__':
    main()
