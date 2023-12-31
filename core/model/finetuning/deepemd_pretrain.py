# -*- coding: utf-8 -*-
"""
@article{DBLP:journals/corr/abs-1812-03664,
  author    = {Han{-}Jia Ye and
               Hexiang Hu and
               De{-}Chuan Zhan and
               Fei Sha},
  title     = {Learning Embedding Adaptation for Few-Shot Learning},
  year      = {2018},
  archivePrefix = {arXiv},
  eprint    = {1812.03664},
}
http://arxiv.org/abs/1812.03664

Adapted from https://github.com/Sha-Lab/FEAT.
"""

import torch
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
from ..metric.proto_net import ProtoLayer
from torch.nn import functional as F


class DeepEMD_Pretrain(FinetuningModel):
    def __init__(
        self, feat_dim, train_num_class, val_num_class, **kwargs
    ):
        super(DeepEMD_Pretrain, self).__init__(**kwargs)
        self.train_num_class = train_num_class
        self.val_num_class = val_num_class
        self.feat_dim = feat_dim

        self.train_classifier = nn.Linear(self.feat_dim, self.train_num_class)
        self.val_classifier = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()

        self.encoder = self.emb_func

    def set_forward(self, batch):
        # FIXME:  do not do validation in first 500 epoches # # test on 16-way 1-shot
        """
        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        
        # # new
        # feat = feat.view(feat.size(0), -1)

        # print("feat.shape")
        # print(feat.shape)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output = self.val_classifier(
            query_feat,
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num,
            mode="euclidean",
        ).reshape(-1, self.way_num)

        acc = accuracy(output, query_target.reshape(-1))

        return output, acc
    
    def encode(self, x, dense=True):

        # print("x:")
        # print(x)

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch,
                          x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x

        else:
            x = self.encoder(x)
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.args.feature_pyramid is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        feature_pyramid = [int(size)
                           for size in self.args.feature_pyramid.split(',')]
        for size in feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(
                feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(
            feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out

    def set_forward_loss(self, batch):
        """
        :param batch:
        :return:
        """
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        # 标准的分类损失
        logits = self.train_classifier(self.encode(image, dense=False).squeeze(-1).squeeze(-1))

        loss = F.cross_entropy(logits, target)

        acc = accuracy(logits, target)
        
        return logits, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        raise NotImplementedError
