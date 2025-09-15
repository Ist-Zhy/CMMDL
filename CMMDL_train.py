import time
import logging
import os
from Net.CMMDL import CMMDL
from logger import setup_logger
from LossFunsion import fusion_loss
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset import H5Dataset
import random
import numpy as np

import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# torchrun --nnodes=1 --nproc_per_node 2 CMMDL_train.py
def init_parser():
    parser = argparse.ArgumentParser(description='Image Fusion')
    parser.add_argument('--lr_start', default=1e-4, help='Initial learning rate')
    parser.add_argument('--optim_step', default=20, type=int, help='Learning rate decay')
    parser.add_argument('--optim_gamma', default=0.5, help='decay factor')
    parser.add_argument('--number_epoch', default=100, help='epoch')
    parser.add_argument('--batch_size', default=4, help='batch_size')
    parser.add_argument('--save_model_path', default='./Model', help='model path')
    parser.add_argument('--dataset', default='data/MSRS_train_imgsize_128_stride_200.h5', help='dataset path')
    parser.add_argument('--num_workers', default=8, help='num_workers')
    arg = parser.parse_args()
    return arg

def train_fusion(args_confing, logger_):
    print(args_confing.save_model_path)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device_id = rank % torch.cuda.device_count()
    if not os.path.exists(args_confing.save_model_path):
        os.makedirs(args_confing.save_model_path)
    fusion_model = CMMDL().to(device_id)
    fusion_model = DDP(module=fusion_model, device_ids=[device_id], find_unused_parameters=True)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args_confing.lr_start)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args_confing.optim_step,
                                                gamma=args_confing.optim_gamma)
    train_dataset = H5Dataset(args_confing.dataset)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    train_dataset = DataLoader(train_dataset,
                         batch_size=args_confing.batch_size,
                         pin_memory=True,
                         sampler=train_sampler,
                         num_workers=args_confing.num_workers)
    train_data_len = len(train_dataset)
    criteria_fusion = fusion_loss()
    st = glob_st = time.time()
    for epoch in range(args_confing.number_epoch):
        for i, (data_VIS, data_IR, Mask) in enumerate(train_dataset):
            data_VIS, data_IR, Mask = data_VIS.cuda(), data_IR.cuda(), Mask.cuda()
            fusion_model.train()
            data_fuse = fusion_model(data_VIS, data_IR)
            optimizer.zero_grad()
            loss_total, loss_grad, loss_int = criteria_fusion(
                data_VIS, data_IR, data_fuse, Mask
            )
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            now_it = train_data_len * epoch + i + 1
            t_intv, glob_t_intv = ed - st, ed - glob_st
            if now_it % 10 == 0:
                if rank == 0:
                    msg = ', '.join(
                        [
                            'epoch: {epoch}',
                            'step: {it}/{max_it}',
                            'loss_total: {loss_total:.4f}',
                            'loss_int: {loss_int:.4f}',
                            'loss_grad: {loss_grad:.4f}',
                            'time: {time:.4f}',
                            'lr:{lr:.6f}'
                        ]
                    ).format(
                        epoch=epoch,
                        it=now_it,
                        max_it=train_data_len * args.number_epoch,
                        loss_total=loss_total.item(),
                        loss_int=loss_int.item(),
                        loss_grad=loss_grad.item(),
                        time=t_intv,
                        lr=optimizer.param_groups[0]['lr']
                    )
                    logger_.info(msg)
        scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6
        if epoch % 10 == 0:
            fusion_model_file = os.path.join(args_confing.save_model_path, f'CMMDL_model_{epoch}.pth')
            torch.save(fusion_model.state_dict(), fusion_model_file)
    fusion_model_file = os.path.join(args_confing.save_model_path, f'CMMDL_model_finally.pth')
    torch.save(fusion_model.state_dict(), fusion_model_file)

def set_seed(seed_):
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)

if __name__ == "__main__":
    seed = 42 # 42
    set_seed(42)
    logger = logging.getLogger()
    log_path = './CMMDL_logs'
    args = init_parser()
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    setup_logger(log_path)
    train_fusion(args, logger)
    print("Train Fusion Model Sucessfully~!")
    print("training Done!")