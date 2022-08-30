from email.mime import base
from math import hypot
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base.conv4d import CenterPivotConv4d as Conv4d

from .ASPP import ASPP
from .chATT import SpatialAttention
from .base.correlation import Correlation
import cv2
import numpy as np

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        #Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

        self.aspp = ASPP(256)

        self.reduce_dim = nn.Sequential(
            nn.Conv2d(577, 256, kernel_size=1, padding=0,bias=False),
            nn.ReLU(inplace=True)
        )

        self.res1_meta = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 2, kernel_size=1))
        
        # self.chATT = ChannelGate(256)
        self.attn = SpatialAttention(256)
        
        self.down_supp = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25))
        
        self.down_query = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25))
        
        self.hyper_final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1,padding='same'), #30
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
    # def mask_feature(self, features, support_mask):
    
    #     for idx, feature in enumerate(features):
    #         mask = F.interpolate(support_mask.float(), feature.size()[2:], mode='bilinear', align_corners=True)
    #         features[idx] = features[idx] * mask
    #     return features
    
    def mask_feature(self, features, support_mask):#bchw
        bs=features[0].shape[0]
        initSize=((features[0].shape[-1])*2,)*2
        support_mask = (support_mask).float()
        support_mask = F.interpolate(support_mask, initSize, mode='bilinear', align_corners=True)
        for idx, feature in enumerate(features):
            feat=[]
            if support_mask.shape[-1]!=feature.shape[-1]:
                support_mask = F.interpolate(support_mask, feature.size()[2:], mode='bilinear', align_corners=True)
            for i in range(bs):
                featI=feature[i].flatten(start_dim=1)#c,hw
                maskI=support_mask[i].flatten(start_dim=1)#hw
                featI = featI * maskI
                maskI=maskI.squeeze()
                meanVal=maskI[maskI>0].mean()
                realSupI=featI[:,maskI>=meanVal]
                if maskI.sum()==0:
                    realSupI=torch.zeros(featI.shape[0],1).cuda()
                feat.append(realSupI)#[b,]ch,w
            features[idx] = feat#nfeatures ,bs,ch,w
        return features
    
    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, query_feats, support_feats, support_mask, stack_ids, nshot):#hypercorr_pyramid, prior, mAP, attn):
        
        query_feat2 = query_feats[2]
        query_feat1 = F.interpolate(query_feats[1], (query_feats[2].size()[2], query_feats[2].size()[3]), mode='bilinear', align_corners=True)
        concat_query_feat = torch.cat([query_feat1, query_feat2], dim=1)
        concat_query_feat = self.down_query(concat_query_feat)
        
        mAPs = []
        priors = []
        corrs = []
        attns = 0
        for i in range(nshot):
            # Caculate MAP
            supp_feat2 = support_feats[i][2]
            supp_feat1 = F.interpolate(support_feats[i][1], (support_feats[i][2].size()[2], support_feats[i][2].size()[3]), mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat1, supp_feat2], dim=1)
            supp_feat = self.down_supp(supp_feat)
            mask = F.interpolate(support_mask[i].unsqueeze(1).cuda(), (support_feats[i][2].size()[2], support_feats[i][2].size()[3]), mode='bilinear', align_corners=True)
            mAP = Weighted_GAP(supp_feat, mask).expand_as(supp_feat)

            mAPs.append(mAP)

            # Attention
            attn = self.attn(supp_feat, support_mask[i].cuda())
            attns += attn

            # prior
            resize_size = support_feats[i][0].size(2)
            tmp_mask = F.interpolate(support_mask[i].unsqueeze(dim=1).cuda(), size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp = support_feats[i][0].clamp(min=0) * tmp_mask
            q = query_feats[0].clamp(min=0)
            s = tmp
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + 1e-7)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + 1e-7)
            prior = similarity.view(bsize, 1, sp_sz, sp_sz)
            priors.append(prior)


            # corr
            support_feat_coor = self.mask_feature(support_feats[i], support_mask[i].unsqueeze(1).clone().cuda())
            corr = Correlation.multilayer_correlation(query_feats, support_feat_coor, stack_ids)

            hypercorr_sqz4 = self.encoder_layer4(corr[0])
            hypercorr_sqz3 = self.encoder_layer3(corr[1])
            hypercorr_sqz2 = self.encoder_layer2(corr[2])
            # Propagate encoded 4D-tensor (Mixing building blocks)
            hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
            hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
            hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

            bsz, ch, ha, wa, hb, wb = hypercorr_mix43.size()
            out_encoded2 = hypercorr_mix43.view(bsz, ch, ha, wa, -1).mean(dim=-1)

            hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
            hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
            hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

            bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
            hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)
            
            corrs.append(hypercorr_encoded)

        # attns = attns/nshot
        # corrs_shot = [corrs[0][i] for i in range(3)]
        # for ly in range(3):
        #     for s in range(1, nshot):
        #         corrs_shot[ly] +=(corrs[s][ly])
        
        # hyper_4 = corrs_shot[0] / nshot
        # hyper_3 = corrs_shot[1] / nshot
        # hyper_2 = corrs_shot[2] / nshot


        # hyper_4 = F.interpolate(hyper_4, (64,64), mode='bilinear', align_corners=True)
        # hyper_3 = F.interpolate(hyper_3, (64,64), mode='bilinear', align_corners=True)
        # hyper_2 = F.interpolate(hyper_2, (64,64), mode='bilinear', align_corners=True)
        # final_corr_encoded = torch.cat([hyper_2, hyper_3, hyper_4], dim=1)
        final_corr_encoded = corrs[0]
        for i in range(1, len(corrs)):
            final_corr_encoded += corrs[i]
        final_corr_encoded = final_corr_encoded/len(corrs)

        final_corr_encoded = self.hyper_final(final_corr_encoded)
        final_corr_encoded = F.interpolate(final_corr_encoded, (64,64), mode='bilinear', align_corners=True)

        priors = torch.cat(priors, dim=1)
        priors = F.interpolate(priors, (64,64), mode='bilinear', align_corners=True)

        concat_query_feat = F.interpolate(concat_query_feat, (64,64), mode='bilinear', align_corners=True)
        
        mAPs = torch.cat(mAPs, 2).sum(2, True)
        mAPs = F.interpolate(mAPs, (64,64), mode='bilinear', align_corners=True)
        
        attns = F.interpolate(attns, (64,64), mode='bilinear', align_corners=True)
        
        encoded = torch.cat([concat_query_feat, mAPs, final_corr_encoded, priors], dim=1)
        encoded = self.reduce_dim(encoded)

        out1 = self.aspp(encoded)
        out1 = self.res1_meta(out1)
        out1 = self.res2_meta(out1) + out1
        meta_out = self.cls_meta(out1)

        return meta_out


# class HPNLearner(nn.Module):
#     def __init__(self, inch):
#         super(HPNLearner, self).__init__()

#         def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
#             assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

#             building_block_layers = []
#             for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
#                 inch = in_channel if idx == 0 else out_channels[idx - 1]
#                 ksz4d = (ksz,) * 4
#                 str4d = (1, 1) + (stride,) * 2
#                 pad4d = (ksz // 2,) * 4

#                 building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
#                 building_block_layers.append(nn.GroupNorm(group, outch))
#                 building_block_layers.append(nn.ReLU(inplace=True))

#             return nn.Sequential(*building_block_layers)

#         outch1, outch2, outch3 = 16, 64, 128

#         # Squeezing building blocks
#         self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
#         self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
#         self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

#         # Mixing building blocks
#         self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
#         self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

#         #Decoder layers
#         # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
#         #                               nn.ReLU(),
#         #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
#         #                               nn.ReLU())

#         # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
#         #                               nn.ReLU(),
#         #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

#         self.aspp = ASPP(256)

#         self.reduce_dim = nn.Sequential(
#             nn.Conv2d(833, 256, kernel_size=1, padding=0,bias=False),
#             nn.ReLU(inplace=True)
#         )

#         self.res1_meta = nn.Sequential(
#             nn.Conv2d(1280, 256, kernel_size=1, padding=0, bias=False),
#             nn.ReLU(inplace=True))
#         self.res2_meta = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True))
#         self.cls_meta = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(256, 2, kernel_size=1))
        
#         # self.chATT = ChannelGate(256)
#         self.attn = SpatialAttention(256)
        
#         self.down_supp = nn.Sequential(
#             nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.25))
        
#         self.down_query = nn.Sequential(
#             nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.25))
        
#         self.hyper_final = nn.Sequential(
#             nn.Conv2d(30, 64, kernel_size=1,padding='same'),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
        


#     def mask_feature(self, features, support_mask):
    
#         for idx, feature in enumerate(features):
#             mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
#             features[idx] = features[idx] * mask
#         return features

#     def interpolate_support_dims(self, hypercorr, spatial_size=None):
#         bsz, ch, ha, wa, hb, wb = hypercorr.size()
#         hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
#         hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
#         o_hb, o_wb = spatial_size
#         hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
#         return hypercorr

#     def forward(self, query_feats, support_feats, support_mask, stack_ids):#hypercorr_pyramid, prior, mAP, attn):
        
#         query_feat2 = query_feats[2]
#         query_feat1 = F.interpolate(query_feats[1], (query_feats[2].size()[2], query_feats[2].size()[3]), mode='bilinear', align_corners=True)
#         concat_query_feat = torch.cat([query_feat1, query_feat2], dim=1)
#         concat_query_feat = self.down_query(concat_query_feat)
        
        
        
#         # Caculate MAP
#         supp_feat2 = support_feats[2]
#         supp_feat1 = F.interpolate(support_feats[1], (support_feats[2].size()[2], support_feats[2].size()[3]), mode='bilinear', align_corners=True)
#         supp_feat = torch.cat([supp_feat1, supp_feat2], dim=1)
#         supp_feat = self.down_supp(supp_feat)
#         mask = F.interpolate(support_mask.unsqueeze(1), (support_feats[2].size()[2], support_feats[2].size()[3]), mode='bilinear', align_corners=True)
#         mAP = Weighted_GAP(supp_feat, mask).expand_as(supp_feat)

#         # Attention
#         attn = self.attn(supp_feat, support_mask)

#         # prior
#         resize_size = support_feats[1].size(2)
#         tmp_mask = F.interpolate(support_mask.unsqueeze(dim=1), size=(resize_size, resize_size), mode='bilinear', align_corners=True)
#         # tmp_mask1 = F.interpolate(support_mask.unsqueeze(dim=1), size=(256, 256), mode='bilinear', align_corners=True)
#         # cv2.imwrite('111/mask.png', np.array((tmp_mask*255).squeeze().cpu()))

#         tmp = support_feats[1].clamp(min=0) * tmp_mask
#         q = query_feats[1].clamp(min=0)
#         s = tmp
#         bsize, ch_sz, sp_sz, _ = q.size()[:]

#         tmp_query = q
#         tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
#         tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

#         tmp_supp = s               
#         tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
#         tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
#         tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

#         similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + 1e-7)   
#         similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
#         similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + 1e-7)
#         prior = similarity.view(bsize, 1, sp_sz, sp_sz)
#         # prior1 = F.interpolate(prior, (32,32), mode='bilinear', align_corners=True)
#         # # prior1[prior1>=0.5] = 1
#         # # prior1[prior1<0.5] = 0
#         # prior1 = prior1*255

#         cv2.imwrite('111/hehe2.png', np.array(prior1.squeeze().cpu()))

#         # corr
#         support_feats = self.mask_feature(support_feats, support_mask)
#         corr = Correlation.multilayer_correlation_hsnet(query_feats, support_feats, stack_ids)



#         # # Encode hypercorrelations from each layer (Squeezing building blocks)
#         # hypercorr_sqz4 = self.encoder_layer4(corr[0])
#         # hypercorr_sqz3 = self.encoder_layer3(corr[1])
#         # hypercorr_sqz2 = self.encoder_layer2(corr[2])

#         # bsz, ch, ha, wa, hb, wb = hypercorr_sqz4.size()
#         # out_encoded1 = hypercorr_sqz4.view(bsz, ch, ha, wa, -1).mean(dim=-1)

#         # # Propagate encoded 4D-tensor (Mixing building blocks)
#         # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
#         # hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
#         # hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

#         # bsz, ch, ha, wa, hb, wb = hypercorr_mix43.size()
#         # out_encoded2 = hypercorr_mix43.view(bsz, ch, ha, wa, -1).mean(dim=-1)

#         # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
#         # hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
#         # hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

#         # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
#         # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)


#         # # Decode the encoded 4D-tensor
#         # hypercorr_decoded = self.decoder1(hypercorr_encoded)
#         # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
#         # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
#         # logit_mask = self.decoder2(hypercorr_decoded)
#         # bsz, ch, ha, wa, hb, wb = corr[0].size()
#         out_encoded1 = corr[0]#.view(bsz, ch, ha, wa, -1).mean(dim=-1)
#         # bsz, ch, ha, wa, hb, wb = corr[1].size()
#         out_encoded2 = corr[1]#.view(bsz, ch, ha, wa, -1).mean(dim=-1)
#         # bsz, ch, ha, wa, hb, wb = corr[2].size()
#         hypercorr_encoded = corr[2]#.view(bsz, ch, ha, wa, -1).mean(dim=-1)


#         out_encoded1 = F.interpolate(out_encoded1, (64,64), mode='bilinear', align_corners=True)
#         out_encoded2 = F.interpolate(out_encoded2, (64,64), mode='bilinear', align_corners=True)
#         hypercorr_encoded = F.interpolate(hypercorr_encoded, (64,64), mode='bilinear', align_corners=True)
#         final_corr_encoded = torch.cat([hypercorr_encoded, out_encoded2, out_encoded1], dim=1)
#         final_corr_encoded = self.hyper_final(final_corr_encoded)
#         prior = F.interpolate(prior, (64,64), mode='bilinear', align_corners=True)
#         concat_query_feat = F.interpolate(concat_query_feat, (64,64), mode='bilinear', align_corners=True)
#         mAP = F.interpolate(mAP, (64,64), mode='bilinear', align_corners=True)
#         attn = F.interpolate(attn, (64,64), mode='bilinear', align_corners=True)
#         encoded = torch.cat([concat_query_feat, mAP, final_corr_encoded, prior, attn], dim=1)
#         encoded = self.reduce_dim(encoded)

#         out1 = self.aspp(encoded)
#         out1 = self.res1_meta(out1)
#         out1 = self.res2_meta(out1) + out1
#         meta_out = self.cls_meta(out1)

#         return meta_out