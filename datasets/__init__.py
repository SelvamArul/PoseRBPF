# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from .imdb import imdb
from .ycb_object import YCBObject

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..')
