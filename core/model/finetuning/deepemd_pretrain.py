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

import argparse
import torch
from torch import nn

from core.utils import accuracy
from .finetuning_model import FinetuningModel
# from ..metric.proto_net import ProtoLayer
from ..metric.deepemd import DeepEMDLayer
from torch.nn import functional as F


class DeepEMD_Pretrain(FinetuningModel):
    def __init__(
        self, mode, feat_dim, train_num_class, val_num_class, args, **kwargs
    ):
        super(DeepEMD_Pretrain, self).__init__(**kwargs)
        self.train_num_class = train_num_class
        self.val_num_class = val_num_class
        self.feat_dim = feat_dim

        self.train_classifier = nn.Linear(self.feat_dim, self.train_num_class)

        self.encoder = self.emb_func

        # 解析参数，这里偷懒了，直接用了argparse 呃呃
        self.args = argparse.Namespace()
        self.args.dataset = args.get('dataset', 'miniimagenet')
        # self.args.data_dir = args.get('data_dir', DATA_DIR)
        self.args.set_val = args.get('set', 'val')

        # about training
        self.args.bs = args.get('bs', 1)
        self.args.max_epoch = args.get('max_epoch', 100)
        self.args.lr = args.get('lr', 0.0005)
        self.args.temperature = args.get('temperature', 12.5)
        self.args.step_size = args.get('step_size', 10)
        self.args.gamma = args.get('gamma', 0.5)
        self.args.val_frequency = args.get('val_frequency', 50)
        self.args.random_val_task = args.get('random_val_task', False)
        self.args.save_all = args.get('save_all', False)

        # about task
        self.args.way = args.get('way', 5)
        self.args.shot = args.get('shot', 1)
        self.args.query = args.get('query', 15)
        self.args.val_episode = args.get('val_episode', 1000)
        self.args.test_episode = args.get('test_episode', 5000)

        # about model
        # self.args.pretrain_dir = args.get('pretrain_dir', PRETRAIN_DIR)
        self.args.metric = args.get('metric', 'cosine')
        self.args.norm = args.get('norm', 'center')
        self.args.deepemd = args.get('deepemd', 'fcn')
        self.args.feature_pyramid = args.get('feature_pyramid', None)
        self.args.num_patch = args.get('num_patch', 9)
        self.args.patch_list = args.get('patch_list', '2,3')
        self.args.patch_ratio = args.get('patch_ratio', 2)

        # solver about
        self.args.solver = args.get('solver', 'opencv')
        self.args.form = args.get('form', 'L2')
        self.args.l2_strength = args.get('l2_strength', 0.000001)

        # SFC
        self.args.sfc_lr = args.get('sfc_lr', 0.1)
        self.args.sfc_wd = args.get('sfc_wd', 0)
        self.args.sfc_update_step = args.get('sfc_update_step', 100)
        self.args.sfc_bs = args.get('sfc_bs', 4)

        # OTHERS
        self.args.gpu = args.get('gpu', '0,1')
        self.args.extra_dir = args.get('extra_dir', None)
        self.args.seed = args.get('seed', 1)

        # add
        self.args.num_classes = args.get('num_classes', 64)
        self.args.way = args.get('way', 5)
        self.args.shot = args.get('shot', 1)
        self.args.query = args.get('query', 16)
        self.args.pretrain = args.get('pretrain', 'origin')

        print('self.args.pretrain:', self.args.pretrain)

        self.mode = mode

        self.deepemd_layer = DeepEMDLayer(
            args=self.args, mode=self.mode, resnet12emd=self.emb_func)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """
        :param batch:
        :return:
        """
        data, global_target = batch
        # print('global_targey:', global_target)

        # 打印data的shape 5*(1+16)张3,84,84 C H W
        # print('data.shape:', data.shape)

        # key: 重新排列数组
        # 重排数据
        new_data = torch.empty((data.size(0), data.size(
            1), data.size(2), data.size(3)), dtype=data.dtype)

        # 重新排列数据
        for i in range(self.args.way):
            for j in range(self.args.shot + self.args.query):
                if j < self.args.shot:
                    new_index = j * self.args.way + i
                else:
                    new_index = self.args.shot * self.args.way + \
                        (j - self.args.shot) * self.args.way + i
                new_data[new_index] = data[i *
                                           (self.args.shot + self.args.query) + j]

        data = new_data.to(self.device)

        k = self.args.way * self.args.shot

        self.deepemd_layer.mode = 'encoder'
        data = self.deepemd_layer(data)

        # print('batch.image:', batch.image)
        # print('batch.image.shape:', batch.image.shape)
        # print('batch.label:', batch.label)
        # print('batch.label.shape:', batch.label.shape)

        data_shot, data_query = data[:k], data[k:]
        self.deepemd_layer.mode = 'meta'
        if self.args.shot > 1:
            data_shot = self.deepemd_layer.get_sfc(data_shot)

        num_gpu = 1  # tag:先固定了
        logits = self.deepemd_layer(
            (data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))

        # 012340123401234...
        label = torch.arange(
            self.args.way, dtype=torch.int8).repeat(self.args.query)
        label = label.type(torch.LongTensor)
        label = label.to(self.device)

        acc = accuracy(logits, label)

        # print('acc:', acc)

        return logits, acc

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
        logits = self.train_classifier(self.encode(
            image, dense=False).squeeze(-1).squeeze(-1))

        loss = F.cross_entropy(logits, target)

        acc = accuracy(logits, target)

        return logits, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        raise NotImplementedError
