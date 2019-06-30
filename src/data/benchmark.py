import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')

        self.ext = ('', '.png')

# class Benchmark(srdata.SRData):
#     def __init__(self, args, name='', train=True, benchmark=True):
#         super(Benchmark, self).__init__(
#                 args, name=name, train=train, benchmark=True
#         )
#
#     def _scan(self):
#         list_hr = []
#         list_lr = [[] for _ in self.scale]
#         for entry in os.scandir(self.dir_hr):
#             filename = os.path.splitext(entry.name)[0]
#             list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
#             new_name = filename
#             for si, s in enumerate(self.scale):
#                 list_lr[si].append(os.path.join(
#                     self.dir_lr,
#                     'X{}/{}x{}{}'.format(s, new_name.replace('_HR_x{}'.format(s),'_LRBI_'), s, self.ext)
#                 ))
#                 # list_lr[si].append(os.path.join(
#                 #     self.dir_lr,
#                 #     'X{}/{}x{}{}'.format(s, filename, s, self.ext)
#                 # ))
#
#         list_hr.sort()
#         for l in list_lr:
#             l.sort()
#
#         return list_hr, list_lr
#
#     def _set_filesystem(self, dir_data):
#         self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
#         self.dir_hr = os.path.join(self.apath, 'HR')
#         self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
#         self.ext = '.png'