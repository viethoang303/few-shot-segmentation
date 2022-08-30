"""Training Script"""
import os
from pickletools import optimize
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from model.hsnet import HypercorrSqueezeNetwork
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, poly_learning_rate
from config import ex

from data_loader.dataloader2 import *
from metrics.metric import compute_dice, compute_iou
from util.metric import Metric

from common.evaluation import Evaluator
from common.logger import Logger, AverageMeter
import cv2


from torch.utils.tensorboard import SummaryWriter

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    writer = SummaryWriter('runs/try_simi_new_SGD')#final_1shot_4C_iou_metric+++')
    _log.info('###### Create model ######')
    model = HypercorrSqueezeNetwork('resnet101', False)
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])

    model.train()


    _log.info('###### Load data ######')
    

    DataModule = UETDataModule('data/uet', batch_size=16, num_workers=4, image_size=256, k_shots=1)
    DataModule.setup()
    trainloader = DataModule.train_dataloader()
    val_loader = DataModule.val_dataloader()


    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay= 0.0001)
    # optimizer = model.module.get_optim(model, lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    i_iter = 0
    val_iter = 0
    min_val_loss = np.Inf
    min_val_iou = 0
    log_loss = {'loss': 0}
    val_log_loss = {'loss': 0}
    _log.info('###### Training ######')
    for epoch in range(200):
        
        training_loss = 0
        val_loss = 0
        val_iou = 0
        model.module.train_mode()

        print('Epoch', epoch, ':')
        for idx, sample_batched in enumerate(trainloader):
            i_iter = i_iter+1
            current_iter = epoch * len(trainloader) + idx + 1

            # poly_learning_rate(optimizer, 1e-6, current_iter, max_iter=200*len(trainloader), power=0.9, index_split=-1, warmup=False, warmup_step=len(trainloader)//2)

            support_images = [[shot.cuda() for shot in way]
                            for way in sample_batched['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]
            support_bg_mask = [[1-shot.float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]

            query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.cuda() for query_label in sample_batched['query_labels']], dim=0)
            
            support_images = torch.cat([torch.cat(way, dim=0) for way in support_images])
            support_fg_mask = torch.stack([torch.stack(way, dim=0) for way in support_fg_mask]).squeeze(0).squeeze(0)
            query_images = sample_batched['query_images'].cuda()
            query_labels = sample_batched['query_labels'].long().cuda()
            # labels = sample_batched['query_labels']
            # labels = F.interpolate(labels.unsqueeze(1), (256,256), mode='bilinear', align_corners=True)
            # cv2.imwrite('111/gt1.png', np.array((labels*255).squeeze().cpu()))

            # Forward and Backward
            optimizer.zero_grad()
            # query_pred1, query_pred2, query_pred3 = model.module.predict_mask_nshot(sample_batched, nshot=5)
            # logit_mask = model(query_images, support_images, support_fg_mask)
            
            logit_mask = model.module.forward_nshot(sample_batched, nshot=1)

            query_loss = criterion(logit_mask, query_labels)
            


            loss = query_loss
            training_loss += loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Log loss
            query_loss = loss.detach().data.cpu().numpy()
            _run.log_scalar('loss', query_loss)
            log_loss['loss'] += query_loss


            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:
                loss = log_loss['loss'] / (i_iter + 1)
                print(f'step {i_iter+1}: loss: {loss}')

                writer.add_scalar('Train step loss',
                            loss,
                            i_iter + 1)
        
        print("Training loss:", training_loss/len(trainloader))
        writer.add_scalar('Training loss',
                            training_loss/len(trainloader),
                            epoch)
        

        # Testing time
        metric = Metric(max_label=3, n_runs=1)
        print('VALIDATION TIME epoch', epoch, ':')
        model.eval()
        with torch.no_grad():
            for idx, sample_batched in enumerate(val_loader):
                label_ids = list(sample_batched['class_ids'])
                val_iter += 1

                support_images = [[shot.cuda() for shot in way]
                            for way in sample_batched['support_images']]
                support_fg_mask = [[shot.float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]
                support_bg_mask = [[1-shot.float().cuda() for shot in way]
                            for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                            for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
            

                support_images = torch.cat([torch.cat(way, dim=0) for way in support_images])
                support_fg_mask = torch.stack([torch.stack(way, dim=0) for way in support_fg_mask], dim=0).squeeze(0).squeeze(0)
                query_images = sample_batched['query_images'].cuda()
                query_labels = sample_batched['query_labels'].long().cuda()
                
                # _,_,query_pred = model.module.predict_mask_nshot(sample_batched, nshot=5)
                # logit_mask = model(query_images, support_images, support_fg_mask)
                # pred_mask = logit_mask.argmax(dim=1)
                logit_mask = model.module.forward_nshot(sample_batched, nshot=1)
                # query_loss = model.module.forward_nshot(sample_batched, nshot=5, train=False)
                query_loss = criterion(logit_mask, query_labels)
                loss = query_loss
                val_loss+=loss

                metric.record(np.array(logit_mask.argmax(dim=1).cpu()),
                              np.array(query_labels.cpu()),
                              labels=label_ids, n_run=1)



                # Log val loss step
                loss = loss.detach().data.cpu().numpy()
                _run.log_scalar('val_loss', loss)
                val_log_loss['loss'] += loss

                # Save step model
                if (val_iter + 1) % _config['print_interval'] == 0:
                    loss = val_log_loss['loss'] / (val_iter + 1)
                    print(f'step {val_iter+1}: loss: {loss}')

                    writer.add_scalar('Validation step loss',
                            loss,
                            val_iter + 1)
        labels = [2,3]
        classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
        classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

        print("mean IoU:", meanIoU)

        writer.add_scalar('Validation iou',
                            meanIoU,#val_iou/2000,
                            epoch)

        # print('Validation loss:', val_loss/len(val_loader))
        writer.add_scalar('Validation loss',
                            val_loss/len(val_loader),
                            epoch)

        if meanIoU > min_val_iou:
            min_val_iou = meanIoU
            _log.info('###### Saving final model ######')
            torch.save(model.state_dict(),
                    os.path.join(f'{_run.observers[0].dir}/snapshots', 'final.pth'))


            