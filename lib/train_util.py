import torch
import os
from .mesh_util import *
from .sample_util import *
from PIL import Image
from tqdm import tqdm
import trimesh


def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh_udf(opt, net, cuda, data, save_path, num_steps=10, num_points=900000):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    for param in net.parameters():
        param.requires_grad = False

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = save_path[:-4] + '.png'
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    if opt.anchor:
        verts = reconstruction_anchor(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps, num_points=num_points)

        for i in range(net.hg_pc.size(0)):
            out_file_off = 'anchor_pred_{}.off'.format(i)
            trimesh.Trimesh(vertices=net.hg_pc[i].transpose(0, 1).detach().cpu().numpy(), faces=[]).export(save_path[:-4] + out_file_off)
    else:
        verts = reconstruction(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps)

    trimesh.Trimesh(vertices=verts, faces = []).export(save_path[:-4] + 'dpc_{}.off'.format(num_steps))


def gen_mesh_hd_udf(opt, net, cuda, data, save_path, num_steps=10, num_points=900000):
    image_tensor = data['img'].to(device=cuda)
    image_low_tensor = data['img_low'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    for param in net.parameters():
        param.requires_grad = False

    net.filter_global(image_low_tensor)
    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = save_path[:-4] + '.png'
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    if opt.anchor:
        verts = reconstruction_anchor(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps, num_points=num_points)

        for i in range(net.netG.hg_pc.size(0)):
            out_file_off = 'anchor_pred_{}.off'.format(i)
            trimesh.Trimesh(vertices=net.netG.hg_pc[i].transpose(0, 1).detach().cpu().numpy(), faces=[]).export(save_path[:-4] + out_file_off)
    else:
        verts = reconstruction(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps)

    trimesh.Trimesh(vertices=verts, faces = []).export(save_path[:-4] + 'dpc_{}.off'.format(num_steps))


def gen_mesh_udf_all(opt, net, cuda, data, save_path, save_npz_path, img_name, num_steps=10, num_points=900000):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    for param in net.parameters():
        param.requires_grad = False

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = os.path.join(save_path, img_name + '.png')
    save_off_path = os.path.join(save_path, img_name + '.off')
    save_anchor_path = os.path.join(save_path, img_name + '_anchor.off')
    save_npz_path = os.path.join(save_npz_path, img_name + '.npz')
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    if opt.anchor:
        verts = reconstruction_anchor(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps, num_points=num_points)

        trimesh.Trimesh(vertices=net.hg_pc[0].transpose(0, 1).detach().cpu().numpy(), faces=[]).export(save_anchor_path)
    else:
        verts = reconstruction(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps)

    trimesh.Trimesh(vertices=verts, faces=[]).export(save_off_path)
    np.savez(save_npz_path, points=verts)


def gen_mesh_udf_hd_all(opt, net, cuda, data, save_path, save_npz_path, img_name, num_steps=10, num_points=900000):
    image_tensor = data['img'].to(device=cuda)
    image_low_tensor = data['img_low'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    for param in net.parameters():
        param.requires_grad = False

    net.filter_global(image_low_tensor)
    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = os.path.join(save_path, img_name + '.png')
    save_off_path = os.path.join(save_path, img_name + '.off')
    save_anchor_path = os.path.join(save_path, img_name + '_anchor.off')
    save_npz_path = os.path.join(save_npz_path, img_name + '.npz')
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    if opt.anchor:
        verts = reconstruction_anchor(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps, num_points=num_points)

        trimesh.Trimesh(vertices=net.netG.hg_pc[0].transpose(0, 1).detach().cpu().numpy(), faces=[]).export(save_anchor_path)
    else:
        verts = reconstruction(net, cuda, calib_tensor, b_min, b_max, max_dist=opt.max_dist, filter_val=opt.filter_val, num_steps=num_steps)

    trimesh.Trimesh(vertices=verts, faces=[]).export(save_off_path)
    np.savez(save_npz_path, points=verts)


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)


def calc_error_udf(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

            erorr_arr.append(error.item())

    return np.average(erorr_arr)


def calc_error_anchor(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    # with torch.no_grad():
    error_rec_arr, error_anchor_arr, error_direct_arr, prec_arr, recall_arr = [], [], [], [], []
    for idx in range(num_tests):
        data = dataset[idx * len(dataset) // num_tests]
        # retrieve the data
        image_tensor = data['img'].to(device=cuda)
        calib_tensor = data['calib'].to(device=cuda)
        sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
        if opt.num_views > 1:
            sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
        label_tensor = data['labels'].to(device=cuda).unsqueeze(0)
        kp_tensor = data['key_points'].to(device=cuda).unsqueeze(0)

        if opt.grad_constraint:
            neigh_tensor = data['neighbors'].to(device=cuda).unsqueeze(0)
            res, error_rec, error_anchor, error_direct = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, key_points=kp_tensor, neighbors=neigh_tensor)

            error_direct_arr.append(error_direct.item())
        else:
            res, error_rec, error_anchor = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, key_points=kp_tensor)

        error_rec_arr.append(error_rec.item())
        error_anchor_arr.append(error_anchor.item())

    if opt.grad_constraint:
        return np.average(error_rec_arr), np.average(error_anchor_arr), np.average(error_direct_arr)
    else:
        return np.average(error_rec_arr), np.average(error_anchor_arr)
