import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.train_util import *
from lib.data import *
from lib.model import *

# get options
opt = BaseOptions().parse()

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = TrainDatasetDF3D(opt, phase='train')
    test_dataset = TrainDatasetDF3D(opt, phase='test')

    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    if opt.anchor:
        if opt.backbone_detach:
            netG = AnchorUdfDetachNet(opt, projection_mode).to(device=cuda)
        else:
            netG = AnchorUdfNet(opt, projection_mode).to(device=cuda)
    else:
        netG = UdfNet(opt, projection_mode).to(device=cuda)

    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    lr = opt.learning_rate
    print('Using Network: ', netG.name)
    
    def set_train():
        netG.train()
        if opt.backbone_detach:
            netG.image_filter.eval()
            netG.svr_net.eval()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        netG.load_state_dict(torch.load(model_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    loss_log = os.path.join(opt.checkpoints_path, opt.name, 'loss_log.txt')

    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch,0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()
        mean_error_rec = 0
        mean_error_anchor = 0
        mean_error_direct = 0

        set_train()
        iter_data_time = time.time()

        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)

            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)

            label_tensor = train_data['labels'].to(device=cuda)

            if opt.anchor:
                kp_tensor = train_data['key_points'].to(device=cuda)

                if opt.grad_constraint:
                    neigh_tensor = train_data['neighbors'].to(device=cuda)

                    if opt.backbone_detach:
                        res, error_rec, error_direct = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, key_points=kp_tensor, neighbors=neigh_tensor)
                        error = 0.5 * (error_rec + error_direct * 0.02)
                        mean_error_direct += (error_direct.data.item() / len(train_data_loader))
                    else:
                        res, error_rec, error_anchor, error_direct = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, key_points=kp_tensor, neighbors=neigh_tensor)
                        error = 0.5 * (error_rec + error_anchor + error_direct * 0.02)
                else:
                    res, error_rec, error_anchor = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, key_points=kp_tensor)
                    error = 0.5 * (error_rec + error_anchor)
                    mean_error_anchor += (error_anchor.data.item() / len(train_data_loader))

                mean_error_rec += (error_rec.data.item() / len(train_data_loader))
            else:
                res, error = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)
                mean_error_rec += (error.data.item() / len(train_data_loader))

            optimizerG.zero_grad()
            error.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                if opt.anchor:
                    if opt.grad_constraint:
                        if opt.backbone_detach:
                            message = 'Name: {0} | Epoch: {1} | {2}/{3} | ErrREC: {4:.06f} | ErrDRE: {5:.06f} | dataT: {6:.05f} | netT: {7:.05f} | ETA: {8:02d}:{9:02d}'.format(
                                opt.name, epoch, train_idx, len(train_data_loader), error_rec.item(), error_direct.item(),
                                                                                    iter_start_time - iter_data_time,
                                                                                    iter_net_time - iter_start_time, int(eta // 60),
                                int(eta - 60 * (eta // 60)))
                        else:
                            message = 'Name: {0} | Epoch: {1} | {2}/{3} | ErrREC: {4:.06f} | ErrPC: {5:.06f} | ErrDRE: {6:.06f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                                    opt.name, epoch, train_idx, len(train_data_loader), error_rec.item(), error_anchor.item(), error_direct.item(),
                                                                                        iter_start_time - iter_data_time,
                                                                                        iter_net_time - iter_start_time, int(eta // 60),
                                    int(eta - 60 * (eta // 60)))
                    else:
                        message = 'Name: {0} | Epoch: {1} | {2}/{3} | ErrREC: {4:.06f} | ErrPC: {5:.06f} | LR: {6:.06f} | Sigma: {7:.04f} | dataT: {8:.05f} | netT: {9:.05f} | ETA: {10:02d}:{11:02d}'.format(
                                opt.name, epoch, train_idx, len(train_data_loader), error_rec.item(), error_anchor.item(), lr, opt.sigma[-1],
                                                                                    iter_start_time - iter_data_time,
                                                                                    iter_net_time - iter_start_time, int(eta // 60),
                                int(eta - 60 * (eta // 60)))
                else:
                    message = 'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | Sigma: {6:.04f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                            opt.name, epoch, train_idx, len(train_data_loader), error.item(), lr, opt.sigma[-1],
                                                                                iter_start_time - iter_data_time,
                                                                                iter_net_time - iter_start_time, int(eta // 60),
                            int(eta - 60 * (eta // 60)))

                print(message)
                with open(loss_log, "a") as log_file:
                    log_file.write('%s\n' % message)

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            iter_data_time = time.time()

        torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

        message = 'mean train L1Loss: {0:06f} ChamLoss: {1:06f} DirectLoss: {2:06f}'.format(mean_error_rec, mean_error_anchor, mean_error_direct)
        print(message)
        with open(loss_log, "a") as log_file:
            log_file.write('%s\n' % message)

        #### test
        set_eval()

        if not opt.no_num_eval:
            test_losses = {}
            print('calc error (test) ...')
            if opt.anchor:
                test_errors = calc_error_anchor(opt, netG, cuda, test_dataset, 100)
                if opt.grad_constraint:
                    message = 'eval test L1Loss: {0:06f} ChamLoss: {1:06f} DirectLoss: {2:06f} GradLoss: {3:06f}'.format(*test_errors)
                    L1Loss, ChamLoss, DirectLoss, GradLoss = test_errors
                    test_losses['L1Loss(test)'] = L1Loss
                    test_losses['ChamLoss(test)'] = ChamLoss
                    test_losses['DirectLoss(test)'] = DirectLoss
                    test_losses['GradLoss(test)'] = GradLoss
                else:
                    message = 'eval test L1Loss: {0:06f} ChamLoss: {1:06f}'.format(*test_errors)
                    L1Loss, ChamLoss = test_errors
                    test_losses['L1Loss(test)'] = L1Loss
                    test_losses['ChamLoss(test)'] = ChamLoss
            else:
                test_errors = calc_error_udf(opt, netG, cuda, test_dataset, 100)
                message = 'eval test L1Loss: {0:06f}'.format(test_errors)
                L1Loss = test_errors
                test_losses['L1Loss(test)'] = L1Loss

            print(message)
            with open(loss_log, "a") as log_file:
                log_file.write('%s\n' % message)

            print('calc error (train) ...')
            if opt.anchor:
                train_dataset.is_train = False
                train_errors = calc_error_anchor(opt, netG, cuda, train_dataset, 100)
                train_dataset.is_train = True
                if opt.grad_constraint:
                    message = 'eval train L1Loss: {0:06f} ChamLoss: {1:06f} DirectLoss: {2:06f} GradLoss: {3:06f}'.format(*train_errors)
                    L1Loss, ChamLoss, DirectLoss, GradLoss = train_errors
                    test_losses['L1Loss(train)'] = L1Loss
                    test_losses['ChamLoss(train)'] = ChamLoss
                    test_losses['DirectLoss(train)'] = DirectLoss
                    test_losses['GradLoss(train)'] = GradLoss
                else:
                    message = 'eval train L1Loss: {0:06f} ChamLoss: {1:06f}'.format(*train_errors)
                    L1Loss, ChamLoss = train_errors
                    test_losses['L1Loss(train)'] = L1Loss
                    test_losses['ChamLoss(train)'] = ChamLoss
            else:
                train_dataset.is_train = False
                train_errors = calc_error_udf(opt, netG, cuda, train_dataset, 100)
                train_dataset.is_train = True
                message = 'eval train L1Loss: {0:06f}'.format(train_errors)
                L1Loss = train_errors
                test_losses['L1Loss(train)'] = L1Loss

            print(message)
            with open(loss_log, "a") as log_file:
                log_file.write('%s\n' % message)


if __name__ == '__main__':
    train(opt)