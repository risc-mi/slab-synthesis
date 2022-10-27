#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os

from libs.util.misc import headline
from run.config import get_config, get_result_root
from core.register import generate_random_warps


def main():
    config = get_config()
    root = get_result_root()
    samples_path = os.path.join(root, 'samples')
    warps_path = os.path.join(root, 'warps')
    models = config.models

    headline('Register: dummy script to register samples')
    print("The samples path is set to: {}".format(samples_path))
    print("The warp results path is set to: {}".format(warps_path))

    for model in models:
        print("-- Generating results for model: {}".format(model))
        model_path = os.path.join(warps_path, model)
        generate_random_warps(samples_path, model_path, override=config.override)


if __name__ == '__main__':
    main()
