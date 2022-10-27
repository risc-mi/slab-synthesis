#  Copyright (c) 2022. RISC Software GmbH.
#  All rights reserved.

import json
import os
import pickle
import sys
import traceback
from contextlib import closing
from tarfile import TarFile
from tempfile import TemporaryDirectory
from typing import Union


def mkdirs(*args):
    """
    Creates one or many directories in case they don't exist.
    param *args: arbitrary number of entries, each a path or list of path to the directories to create
    Examples:
        makedirs('./a')
        makedirs('./a', './b')
        makedirs(['./a', './b')
        makedirs(['./a', './b'], './c)
    """
    for arg in args:
        if isinstance(arg, str):
            arg = [arg]
        for p in arg:
            os.makedirs(p, exist_ok=True)


def read_json(path: str, warn=True):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        if warn:
            traceback.print_exc()
            raise RuntimeError("Failed to load json data from file: {}".format(path))
        raise


def write_json(data: Union[dict, list], path: str):
    with open(path, 'w') as f:
        return json.dump(data, f)


def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class SafeTemporaryDirectory(TemporaryDirectory):
    """
    An overload of TemporaryDirectory that does not fail when the directory is not able to cleanup.
    After 3 failed retries cleanup is aborted without raising an Exception.
    """
    def __exit__(self, *args, **kwargs):
        for i in range(3):
            try:
                self.cleanup()
                break
            except:
                print("Cleanup attempt {} of temporary directory failed ({})".format(i+1, self.name), file=sys.stderr)


class TemporaryTarMirror(SafeTemporaryDirectory):
    """
    An overload of SafeTemporaryDirectory which allows the user to treat a .tar file as regular directories by mirroring
    its contents to a temporary directory.
    """
    def __init__(self, zip_path: str, readonly=False, **kwargs):
        super().__init__(**kwargs)
        self.zip_path = zip_path
        self.readonly = readonly
        if os.path.exists(self.zip_path):
            with closing(TarFile.open(zip_path, 'r')) as tar:
                tar.extractall(self.name)

    def cleanup(self):
        if not self.readonly:
            dir_path = os.path.dirname(self.zip_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with closing(TarFile.open(self.zip_path, 'w:gz')) as tar:
                self._tardir(self.name, tar)
        super().cleanup()

    def _tardir(self, path, tar):
        for root, dirs, files in os.walk(path):
            for file in files:
                src = os.path.join(root, file)
                dst = os.path.relpath(os.path.join(root, file), path)
                tar.write(src, dst)