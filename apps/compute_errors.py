import numpy as np
import os
import trimesh
import torch
import argparse
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D


def load_trimesh_df3d(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s.obj' % (sub_name+'.obj_scaled')))

    return meshs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run error computing'
    )
    parser.add_argument('--results_path', type=str, default='./results_eval', help='path to save results')
    parser.add_argument('--root_path', type=str, help='path to dataset root')

    args = parser.parse_args()

    root_path = args.root_path
    OBJ_path = os.path.join(root_path, 'GEO', 'OBJ')
    PARAM_path = os.path.join(root_path, 'PARAM')

    chamL2 = dist_chamfer_3D.chamfer_3DDist()

    var_subjects = list(np.loadtxt(os.path.join(root_path, 'val.txt'), dtype=str))
    yaw_list = [0, 40, 320]

    chamL2_log = os.path.join(args.results_path, 'chamL2_p2s_log.txt')

    dist_mean_list = []
    p2s_mean_list = []

    for subject in var_subjects:
        print(subject)
        with open(chamL2_log, "a") as log_file:
            log_file.write('%s\n' % subject)

        dist_pc_list = []
        dist_p2s_list = []

        result_npz_path = os.path.join(args.results_path, 'npz', subject)
        mesh = trimesh.load(os.path.join(OBJ_path, subject, '%s.obj' % (subject+'.obj_scaled')))

        sample_point_count = 1000000
        surface_point_cloud = mesh.sample(sample_point_count, return_index=False)
        surface_point_cloud = torch.Tensor(surface_point_cloud).float().cuda()

        for vid in yaw_list:
            npz_file = os.path.join(result_npz_path, '%d_%d_%02d.npz' % (vid, 0, 0))
            result_npz = np.load(npz_file)
            result_points = result_npz['points']

            result_points = torch.Tensor(result_points).unsqueeze(0).float().cuda()

            result_points = (result_points + 1) / 2.0

            param_path = os.path.join(PARAM_path, subject, '%d_%d_%02d.npy' % (vid, 0, 0))

            param = np.load(param_path, allow_pickle=True)
            ortho_ratio = param.item().get('ortho_ratio')
            scale = param.item().get('scale')
            center = param.item().get('center')
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(512 // 2)
            uv_intrinsic[1, 1] = 1.0 / float(512 // 2)
            uv_intrinsic[2, 2] = 1.0 / float(512 // 2)
            trans_intrinsic = np.identity(4)
            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))

            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float().cuda()

            rot = calib[:3, :3]
            trans = calib[:3, 3:4]

            proj_gt = torch.matmul(rot, surface_point_cloud.T) + trans
            proj_gt = proj_gt.T.unsqueeze(0)

            proj_gt = (proj_gt + 1) / 2.0

            with torch.no_grad():
                dist_pc1, dist_pc2, _, _ = chamL2(result_points, proj_gt)  # error_pc1 = [B, M], error_pc2 = [B, M]

                dist_pc = (dist_pc1.mean() + dist_pc2.mean()) * 0.5
                dist_pc_list.append(dist_pc)

                dist_p2s_list.append(dist_pc1.mean())

        dist_mean = torch.stack(dist_pc_list).mean()
        dist_mean_list.append(dist_mean)

        p2s_mean = torch.stack(dist_p2s_list).mean()
        p2s_mean_list.append(p2s_mean)

        message = 'Chamer L2: {}'.format(dist_mean.item())
        print(message)
        with open(chamL2_log, "a") as log_file:
            log_file.write('%s\n' % message)

        message = 'P2S L2: {}'.format(p2s_mean.item())
        print(message)
        with open(chamL2_log, "a") as log_file:
            log_file.write('%s\n' % message)

    dist_final = torch.stack(dist_mean_list).mean()
    p2s_final = torch.stack(p2s_mean_list).mean()

    message = 'Final Chamer L2: {}'.format(dist_final.item())
    print(message)
    with open(chamL2_log, "a") as log_file:
        log_file.write('%s\n' % message)

    message = 'Final P2S L2: {}'.format(p2s_final.item())
    print(message)
    with open(chamL2_log, "a") as log_file:
        log_file.write('%s\n' % message)
