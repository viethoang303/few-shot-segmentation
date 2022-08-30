F"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from model.hsnet import *
# from dataloaders.customized import voc_fewshot, coco_fewshot
# from dataloaders.transforms import ToTensorNormalize
# from dataloaders.transforms import Resize, DilateScribble
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex

from data_loader.dataloader2 import *
import cv2

@ex.automain
def main(_run, _config, _log):
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


    _log.info('###### Create model ######')
    model = HypercorrSqueezeNetwork('resnet101', False)
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])

    model.load_state_dict(torch.load('/data.local/all/viethoang/hsnet/runs/PANet_VOC_align_sets_0_1way_1shot_[train]/final_1shot_2C_metric_train/snapshots/final.pth', map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')

    labels = [2,3]

    _log.info('###### Testing begins ######')
    metric = Metric(max_label=3, n_runs=5)
    with torch.no_grad():
        for run in range(5):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            _log.info(f'### Load data ###')
            
            DataModule = UETDataModule('data/uet', batch_size=1, num_workers=2, image_size=256, k_shots=5)
            DataModule.setup()
            testloader = DataModule.val_dataloader()
            _log.info(f"Total # of Data: {len(testloader)}")


            for sample_batched in tqdm.tqdm(testloader):
                label_ids = list(sample_batched['class_ids'])
                support_images = [[shot.cuda() for shot in way]
                                    for way in sample_batched['support_images']]
                support_fg_mask = [[shot.float().cuda() for shot in way]
                                    for way in sample_batched['support_mask']]
                support_bg_mask = [[1-shot.float().cuda() for shot in way]
                                    for way in sample_batched['support_mask']]
                
                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                    [query_label.cuda()for query_label in sample_batched['query_labels']], dim=0)

                
                
                support_images = torch.cat([torch.cat(way, dim=0) for way in support_images])
                support_fg_mask = torch.stack([torch.stack(way, dim=0) for way in support_fg_mask]).squeeze(0).squeeze(0)


                query_images = sample_batched['query_images'].cuda()
                query_labels = sample_batched['query_labels'].long().cuda()
                
                # query_pred = model.module.predict_mask_nshot(sample_batched, nshot=5)
                query_pred = model(query_images, support_images, support_fg_mask)
                # _,_,query_pred = model.module.forward_nshot(sample_batched, nshot=5)

                metric.record(np.array(query_pred.argmax(dim=1).cpu()),
                              np.array(query_labels.cpu()),
                              labels=label_ids, n_run=run)

                # metric.record(np.array(query_pred.cpu()),
                #               np.array(sample_batched['label'].cpu()),
                #               labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')
