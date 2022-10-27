#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

from libs.util.misc import headline
from run.config import get_config, get_resource_root, get_local_dataset_root
from core.importing import import_all


def main():
    config = get_config()
    skip = config.skip
    override = config.override
    input_root = get_local_dataset_root()
    result_root = get_resource_root()

    headline("Decathlon Dataset Importer")
    print("The input root folder is set to: {}".format(input_root))
    print("The result root folder is set to: {}".format(result_root))
    print("Skip tasks: {}".format(', '.join(str(id) for id in sorted(skip)) if skip else 'None'))
    print()

    import_all(input_root, result_root, override)


if __name__ == '__main__':
    main()
