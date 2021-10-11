import numpy as np
import torch
from torch.nn import functional as F


def reconstruction(net, cuda, calib_tensor, b_min, b_max, max_dist=0.1, filter_val=0.006, num_steps=10):
    length = b_max[0] - b_min[0]

    sample_num = 200000
    samples_cpu = np.zeros((0, 3))
    samples = torch.rand(1, sample_num, 3).float().to(device=cuda) * length + b_min[0]

    samples.requires_grad = True

    num_points = 900000

    i = 0
    while len(samples_cpu) < num_points:
        print('iteration', i)

        for j in range(num_steps):
            print('refinement', j)
            net.query(torch.transpose(samples, 1, 2), calib_tensor)
            pred = net.get_preds()
            pred = pred.squeeze(1)
            print(pred)

            df_pred = torch.clamp(pred, max=max_dist)

            df_pred.sum().backward()

            gradient = samples.grad.detach()
            samples = samples.detach()
            df_pred = df_pred.detach()
            samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  # better use Tensor.copy method?
            samples = samples.detach()
            samples.requires_grad = True


        print('finished refinement')

        if not i == 0:
            samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))

        samples = samples[df_pred < 0.03].unsqueeze(0)
        indices = torch.randint(samples.shape[1], (1, sample_num))
        samples = samples[[[0, ] * sample_num], indices]
        samples += (max_dist / 3) * torch.randn(samples.shape).to(device=cuda)  # 3 sigma rule
        samples = samples.detach()
        samples.requires_grad = True

        i += 1
        print(samples_cpu.shape)

    return samples_cpu


def reconstruction_anchor(net, cuda, calib_tensor, b_min, b_max, max_dist=0.1, filter_val=0.006, num_steps=10, num_points=900000):
    length = b_max[0] - b_min[0]

    sample_num = 200000
    samples_cpu = np.zeros((0, 3))
    samples = torch.rand(1, sample_num, 3).float().to(device=cuda) * length + b_min[0]

    samples.requires_grad = True

    i = 0
    while len(samples_cpu) < num_points:
        print('iteration', i)

        for j in range(num_steps):
            print('refinement', j)
            net.query(torch.transpose(samples, 1, 2), calib_tensor)
            pred = net.get_preds()
            pred = pred.squeeze(1)
            print(pred)

            df_pred = torch.clamp(pred, max=max_dist)

            df_pred.sum().backward()

            gradient = samples.grad.detach()
            samples = samples.detach()
            df_pred = df_pred.detach()
            samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  # better use Tensor.copy method?
            samples = samples.detach()
            samples.requires_grad = True


        print('finished refinement')

        if not i == 0:
            samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))

        samples = samples[df_pred < 0.03].unsqueeze(0)
        indices = torch.randint(samples.shape[1], (1, sample_num))
        samples = samples[[[0, ] * sample_num], indices]
        samples += (max_dist / 3) * torch.randn(samples.shape).to(device=cuda)  # 3 sigma rule
        samples = samples.detach()
        samples.requires_grad = True

        i += 1
        print(samples_cpu.shape)

    return samples_cpu


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
