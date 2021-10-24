from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging


log = logging.getLogger('trimesh')
log.setLevel(40)

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s.obj' % (sub_name+'.obj_scaled')))

    return meshs


class TrainDatasetDF3DHD(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER_1024')
        self.MASK = os.path.join(self.root, 'MASK_1024')
        self.PARAM = os.path.join(self.root, 'PARAM_1024')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        self.TARGET = os.path.join(self.root, 'TARGET')
        self.KEY_POINT = os.path.join(self.root, 'KEY_POINT')

        self.B_MIN = np.array([-0.5, -0.5, -0.5])
        self.B_MAX = np.array([0.5, 0.5, 0.5])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.num_key_points = self.opt.num_anchor_points

        self.sample_distribution = np.array(self.opt.sample_distribution)
        self.sample_sigmas = np.array(self.opt.sigma)
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.num_samples = np.rint(self.sample_distribution * self.num_sample_inout).astype(np.uint32)

        self.yaw_list = [0, 8, 16, 24, 32, 40, 48, 56, 64, 288, 296, 304, 312, 320, 328, 336, 344, 352]
        self.pitch_list = [0]
        self.subjects = self.get_subjects()
        print(len(self.subjects))

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.to_tensor_512 = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        if not (self.opt.ndf or self.opt.ndf_pc):
            self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects = os.listdir(self.RENDER)
        var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
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
        pitch = self.pitch_list[pid]

        # The ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        render_512_list = []
        mask_512_list = []
        extrinsic_list = []
        scale_intrinsic_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
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
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask_512 = transforms.Resize(512)(mask)
            mask_512 = transforms.ToTensor()(mask_512).float()
            mask_512_list.append(mask_512)

            render_512 = self.to_tensor_512(render)
            render_512 = mask_512.expand_as(render_512) * render_512

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            render_512_list.append(render_512)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            scale_intrinsic_list.append(scale_intrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'img_low': torch.stack(render_512_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'scale_intrinsic': abs(scale_intrinsic_list[0][0][0]) / float(self.opt.loadSize // 2),
            'mask': torch.stack(mask_list, dim=0),
            'mask_512': torch.stack(mask_512_list, dim=0)
        }

    def select_sampling_method(self, subject, scale_intrinsic):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)

        key_points_path = os.path.join(self.KEY_POINT, subject, 'key_point_{}.npz'.format(self.num_key_points))
        key_points_npz = np.load(key_points_path)
        key_points = key_points_npz['kp']
        key_points = key_points.T
        key_points = torch.Tensor(key_points).float()

        sample_points = []
        distances = []
        sample_neighbors = []

        for i, num in enumerate(self.num_samples):
            if self.opt.grad_constraint:
                boundary_samples_path = os.path.join(self.TARGET, subject, 'boundary_{}_direct_samples.npz'.format(self.sample_sigmas[i]))
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_df = boundary_samples_npz['df']
                boundary_sample_neighbors = boundary_samples_npz['neighbors']
                total_indices = np.arange(boundary_sample_points.shape[0])
                np.random.shuffle(total_indices)
                sample_indices = total_indices[:num]

                sample_points.extend(boundary_sample_points[sample_indices])
                distances.extend(boundary_sample_df[sample_indices])
                sample_neighbors.extend(boundary_sample_neighbors[sample_indices])

            else:
                boundary_samples_path = os.path.join(self.TARGET, subject, 'boundary_{}_samples.npz'.format(self.sample_sigmas[i]))
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_df = boundary_samples_npz['df']
                total_indices = np.arange(boundary_sample_points.shape[0])
                np.random.shuffle(total_indices)
                sample_indices = total_indices[:num]

                sample_points.extend(boundary_sample_points[sample_indices])
                distances.extend(boundary_sample_df[sample_indices])

        assert len(sample_points) == self.num_sample_inout
        assert len(distances) == self.num_sample_inout

        samples = np.array(sample_points).T
        distances = np.array(distances)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(distances).float() * scale_intrinsic

        if self.opt.grad_constraint:
            neighbors = np.array(sample_neighbors).T
            neighbors = torch.Tensor(neighbors).float()
        else:
            neighbors = None

        if neighbors is not None:
            return {
                'samples': samples,
                'labels': labels,
                'key_points': key_points,
                'neighbors': neighbors
            }
        else:
            return {
                'samples': samples,
                'labels': labels,
                'key_points': key_points
            }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=self.opt.random_multiview)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject, res['scale_intrinsic'])
            res.update(sample_data)
        
        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0).copy()
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # # pts = torch.addmm(trans, rot, sample_data['samples'])  # [3, N]
        # # pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        # kps = torch.addmm(trans, rot, sample_data['key_points'])
        # kps = 0.5 * (kps.numpy().T + 1.0) * render_data['img'].size(2)
        # for p in kps:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imwrite('test_calib_kp.jpg', img)
        # # cv2.imshow('test', img)
        # # cv2.waitKey(1)

        return res

    def __getitem__(self, index):
        return self.get_item(index)
