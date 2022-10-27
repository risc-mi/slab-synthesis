#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import glob
import os

import SimpleITK as sitk

from libs.pipeline import enumerator
from libs.util.file import mkdirs, TemporaryTarMirror
from run.config import get_default_extension, get_compression_enabled
from libs.util.misc import warn


def import_task(task_name: str,
                input_root: str,
                result_root: str,
                override):
    ext = get_default_extension()
    compress = get_compression_enabled()

    task, task_target = task_name.split('_') # e.g., Task09_Spleen
    task_nr = int(task.lstrip('Task'))
    sources = {
        'image': os.path.join(input_root, task_name, 'imagesTr'),
        'label': os.path.join(input_root, task_name, 'labelsTr')
    }
    mkdirs(result_root)
    for item in enumerator(sources, ext='nii.gz', prefix=''):
        source_name = item['name']

        # get rid of '.' tumbs files
        if not source_name.startswith('.'):
            name = os.path.splitext(source_name)[0]
            target, nr = name.split('_') # e.g., spleen_2
            nr = int(nr)
            name = 't{:02d}__{}_{:03}'.format(task_nr, task_target.lower(), nr)
            image_path = item['image']
            label_path = item['label']
            image_out = os.path.join(result_root, '{}__image.{}'.format(name, ext))
            label_out = os.path.join(result_root, '{}__label.{}'.format(name, ext))

            msg = "Processing {}...".format(name)
            print(msg, end='')
            if override or not all(os.path.exists(p) for p in [image_out, image_path]):
                image = sitk.ReadImage(image_path)
                label = sitk.ReadImage(label_path)

                sitk.WriteImage(image, image_out, compress)
                sitk.WriteImage(label, label_out, compress)
                print("\r{} SUCCESS".format(msg))
            else:
                print("\r{} SKIP".format(msg))


def import_all(input_root, result_root, override=True, skip_tasks=None):
    """
    Imports all volumes and labels from all tasks at the input root folder.
    The tasks have to be provided as the original .tar files from the download site.
    If, for a given task, there exists a folder next to the .tar file, the data in the folder is used instead
    of extracting the .tar file (as a temporary directory).
    :param input_root: input folder containing the .tar files
    :param result_root: folder to write the imported data to
    :param override: whether to override existing data
    :param skip_tasks: optional task ids (as integers) to skip
    :return:
    """
    skip_tasks = set() if skip_tasks is None else set(skip_tasks)
    try:
        input_pattern = os.path.join(input_root, 'Task*.tar')
        for task_file in glob.glob(input_pattern):
            task_name = os.path.splitext(os.path.basename(task_file))[0]
            task_nr = int(task_name.split('_')[0].lstrip('Task'))
            if task_nr not in skip_tasks:
                alt_root = os.path.join(os.path.dirname(task_file), task_name)
                tar = None
                try:
                    if os.path.exists(alt_root):
                        task_root = alt_root
                    else:
                        print('-- EXTRACTING {}...'.format(task_name), end='')
                        temp_name = '_temp_{}_'.format(task_name.lower())
                        tar = TemporaryTarMirror(task_file, readonly=True, dir=result_root, prefix=temp_name)
                        task_root = tar.__enter__()

                    print('\r-- IMPORTING {}...'.format(task_name))
                    import_task(task_name, task_root, result_root, override=override)
                except:
                    if tar is not None:
                        tar.__exit__()
                    raise
            else:
                print('-- SKIPPING {}'.format(task_name))

    except Exception as ex:
        warn("Import of task data failed: {}".format(ex))
        raise
