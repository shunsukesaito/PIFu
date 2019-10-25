from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import cv2


class EvalDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, root=None):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        if root is not None:
            self.root = root
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.phase = 'val'
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.max_view_angle = 360
        self.interval = 1
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_subjects(self):
        var_file = os.path.join(self.root, 'val.txt')
        if os.path.exists(var_file):
            var_subjects = np.loadtxt(var_file, dtype=str)
            return sorted(list(var_subjects))
        all_subjects = os.listdir(self.RENDER)
        return sorted(list(all_subjects))

    def __len__(self):
        return len(self.subjects) * self.max_view_angle // self.interval

    def get_render(self, subject, num_views, view_id=None, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        # For now we only have pitch = 00. Hard code it here
        pitch = 0
        # Select a random view_id from self.max_view_angle if not given
        if view_id is None:
            view_id = np.random.randint(self.max_view_angle)
        # The ids are an even distribution of num_views around view_id
        view_ids = [(view_id + self.max_view_angle // num_views * offset) % self.max_view_angle
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.max_view_angle, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%02d.npy' % (vid, pitch))
            render_path = os.path.join(self.RENDER, subject, '%d_%02d.jpg' % (vid, pitch))
            mask_path = os.path.join(self.MASK, subject, '%d_%02d.png' % (vid, pitch))

            # loading calibration data
            param = np.load(param_path)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = -scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        try:
            sid = index % len(self.subjects)
            vid = (index // len(self.subjects)) * self.interval
            # name of the subject 'rp_xxxx_xxx'
            subject = self.subjects[sid]
            res = {
                'name': subject,
                'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
                'sid': sid,
                'vid': vid,
            }
            render_data = self.get_render(subject, num_views=self.num_views, view_id=vid,
                                          random_sample=self.opt.random_multiview)
            res.update(render_data)
            return res
        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
