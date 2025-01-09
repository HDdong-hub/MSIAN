from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch
import os
from utils import ang2joint
import pickle as pkl
from os import walk


class Datasets(Dataset):

    def __init__(self, opt, split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = ""
        self.split = split
        self.in_n = opt.input_n
        self.out_n = opt.output_n

        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        seq_len = self.in_n + self.out_n

        if split == 0:
            data_path = self.path_to_data + '/train/'
        elif split == 2:
            data_path = self.path_to_data + '/test/'
        elif split == 1:
            data_path = self.path_to_data + '/validation/'
        files = []
        for (dirpath, dirnames, filenames) in walk(data_path):
            files.extend(filenames)

        skel = np.load('smpl_skeleton.npz')
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()[:, :22]
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            parent[i] = parents[i]
        n = 0

        sample_rate = int(60 // 25)

        for f in files:
            with open(data_path + f, 'rb') as f:
                print('>>> loading {}'.format(f))
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['poses_60Hz']
                for i in range(len(joint_pos)):
                    poses = joint_pos[i]
                    fn = poses.shape[0]
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    poses = poses[:, :-2]

                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
                    self.p3d.append(p3d.cpu().data.numpy())

                    if split == 2:

                        valid_frames = np.arange(0, fn - seq_len + 1)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, opt.skip_rate)

                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]


if __name__ == '__main__':
    from utils.opt import Options

    opt = Options().parse()
    ds = Datasets(opt, split=0)
