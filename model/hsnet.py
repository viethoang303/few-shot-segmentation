r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add
from pickletools import optimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat


class HypercorrSqueezeNetwork(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(HypercorrSqueezeNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # self.chATT = ChannelGate(256)
        # self.down_supp = nn.Sequential(
        #     nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.1))

    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            # support_feats = self.mask_feature(support_feats, support_mask)
            # corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

        logit_mask = self.hpn_learner(query_feats, support_feats, support_mask, self.stack_ids)#corr, corr_query, mAP, attn)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, query_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask

    def mask_feature(self, features, support_mask):

        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def get_optim(self, model, lr=1e-6):
        optimizer = torch.optim.SGD( [{'params': model.module.hpn_learner.parameters()}], 
                                    lr=lr, momentum=0.9, weight_decay= 0.0001)
        return optimizer

    def predict_mask_nshot(self, batch, nshot):
    
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_images'].cuda(), batch['support_images'][0][s_idx].cuda(), batch['support_mask'][0][s_idx].cuda())
            if self.use_original_imgsize:
                org_qry_imsize = tuple([256,256])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
            logit_mask_agg += logit_mask#.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg
        # Average & quantize predictions given threshold (=0.5)
        # bsz = logit_mask_agg_1.size(0)
        # max_vote = logit_mask_agg_1.view(bsz, -1).max(dim=1)[0]
        # max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        # max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        # pred_mask_1 = logit_mask_agg_1.float() / max_vote
        # pred_mask_1[pred_mask_1 < 0.5] = 0.0
        # pred_mask_1[pred_mask_1 >= 0.5] = 1.0
        return logit_mask_agg/nshot

    def forward_nshot(self, batch, nshot=5):
        # Perform multiple prediction given (nshot) number of different support sets
        # loss = 0
        supp_feats = []
        with torch.no_grad():
            query_feats = self.extract_feats(batch['query_images'].cuda(), self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            for s_idx in range(nshot):
                # logit_mask = self(batch['query_images'].cuda(), batch['support_images'][0][s_idx].cuda(), batch['support_mask'][0][s_idx].cuda())
            
                
                supp_feat = self.extract_feats(batch['support_images'][s_idx].cuda(), self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                
                supp_feats.append(supp_feat)

        logit_mask = self.hpn_learner(query_feats, supp_feats, batch['support_mask'], self.stack_ids, nshot=nshot)

        if not self.use_original_imgsize:
            org_qry_imsize = tuple([256,256])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
            
        # loss += self.cross_entropy_loss(logit_mask, batch['query_labels'].long().cuda())
        return logit_mask


    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
