r""" Hypercorrelation Squeeze testing code """
import argparse

import torch.nn.functional as F
import torch.nn as nn
import torch

from model.hsnet import HypercorrSqueezeNetwork
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

from data_loader.dataloader2 import UETDataModule


def test(model, dataloader, nshot):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)

        # query_images = batch['query_images'].cuda()
        # query_labels = batch['query_labels'].long().cuda()

        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)
        # print(pred_mask.shape)

        assert pred_mask.size() == batch['query_labels'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_ids'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    args = parser.parse_args()
    

    DataModule = UETDataModule('data/uet', batch_size=4, num_workers=2, k_shots=1)
    DataModule.setup()
    dataloader_test = DataModule.test_dataloader()

    model = HypercorrSqueezeNetwork('resnet101', False)
    model = nn.DataParallel(model.cuda(), device_ids=[0,])

    model.load_state_dict(torch.load('/data.local/all/viethoang/hsnet/runs/PANet_VOC_align_sets_0_1way_1shot_[train]/trainaspp1shoton4C/snapshots/final.pth', map_location='cpu'))
    model.eval()

    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, nshot=1)
        print(test_miou, ' ', test_fb_iou)
    # Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    # Logger.info('==================== Finished Testing ====================')
