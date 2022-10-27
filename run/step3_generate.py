#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import os

from libs.util.misc import headline
from run.config import get_config, get_result_root
from core.generate import get_generator, generate_samples


def main():
    root = get_result_root()
    config = get_config()
    input_path = os.path.join(root, "pp")
    output_path = os.path.join(root, "samples")

    headline('Decathlon: script for generating synthetic slabs from decathlon data')
    print("The input path is set to: {}".format(input_path))
    print("The model path is set to: {}".format(output_path))

    """
    instantiate the generator, we will further use it to export samples but it can be easily integrated
    into a model training routine, look up the method get_generator to learn about different options 
    the generator provides regarding training/prediction, usage of splits and more
    """
    gen = get_generator(input_path, config)

    """
    export a fixed number of samples to the output directory
    see write_sample to learn about the data exported for each sample
    """
    num_samples = config.sample_count
    generate_samples(gen, count=num_samples, output_dir=output_path, prefix=config.prefix)


if __name__ == '__main__':
    main()
