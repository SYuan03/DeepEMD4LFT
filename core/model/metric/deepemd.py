import argparse
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel
from .emd_utils import *

import numpy as np

# fake args
DATA_DIR = '/home/zhengxiawu/work/dataset'
PRETRAIN_DIR = '/root/autodl-tmp/ml/deepemd_pretrain_model/miniimagenet/resnet12/max_acc.pth'


class DeepEMDLayer(nn.Module):
    def __init__(self, args, resnet12emd, mode='meta'):
        super().__init__()

        self.mode = mode

        self.args = args

        self.encoder = resnet12emd

        print('self.args.pretrain:')
        print(self.args.pretrain)

        if self.args.pretrain == 'origin':
            # 打印encoder的state_dict的keys
            print('encoder.state_dict().keys():',
                  self.encoder.state_dict().keys())
            load_model(self.encoder, PRETRAIN_DIR)

        if self.mode == 'pre_train':
            print('self.mode', self.mode)
            self.fc = nn.Linear(640, self.args.num_classes)

    def forward(self, input):
        if self.mode == 'meta':
            support, query = input
            return self.emd_forward_1shot(support, query)

        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)

        elif self.mode == 'encoder':
            if self.args.deepemd == 'fcn':
                dense = True
            else:
                dense = False
            return self.encode(input, dense)
        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input):
        return self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))

    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)

        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(
                similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(
                similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the proto
        SFC = support.view(self.args.shot, -1, 640,
                           support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD(
            [SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(
                        j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node = weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(
                        1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] = (
                        similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature = (self.args.temperature / num_node)
            logitis = similarity_map.sum(-1).sum(-1) * temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(
                num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(
                1 - similarity_map, weight_1, weight_2, form=self.args.form, l2_strength=self.args.l2_strength)

            logitis = (flows*similarity_map).view(num_query,
                                                  num_proto, flows.shape[-2], flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) * temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def encode(self, x, dense=True):

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


class DeepEMD(MetricModel):
    def __init__(self, args, mode, **kwargs):
        # 调用父类的构造函数
        super(DeepEMD, self).__init__(**kwargs)

        # 解析参数，这里偷懒了，直接用了argparse 呃呃
        self.args = argparse.Namespace()
        self.args.dataset = args.get('dataset', 'miniimagenet')
        self.args.data_dir = args.get('data_dir', DATA_DIR)
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
        self.args.pretrain_dir = args.get('pretrain_dir', PRETRAIN_DIR)
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

        # print('self.args.pretrain:', self.args.pretrain)
        # 使用logger

        self.mode = mode

        self.deepemd_layer = DeepEMDLayer(
            args=self.args, mode=self.mode, resnet12emd=self.emb_func)
        self.loss_func = nn.CrossEntropyLoss()      # tag:不确定是不是这个损失函数

    def set_forward_loss(self, batch):

        data, global_target = batch
        # print('global_targey:', global_target)

        # 打印data的shape 5*(1+16)张3,84,84 C H W
        # print('data.shape:', data.shape)

        # key: 重新排列数组
        # 重排数据
        # new_data = torch.empty((data.size(0), data.size(
        #     1), data.size(2), data.size(3)), dtype=data.dtype)

        # # 重新排列数据
        # for i in range(self.args.way):
        #     for j in range(self.args.shot + self.args.query):
        #         if j < self.args.shot:
        #             new_index = j * self.args.way + i
        #         else:
        #             new_index = self.args.shot * self.args.way + \
        #                 (j - self.args.shot) * self.args.way + i
        #         new_data[new_index] = data[i *
        #                                    (self.args.shot + self.args.query) + j]

        # data = new_data.to(self.device)

        data =data.to(self.device)

        # print('set_forward_loss')
        # print('way:', self.args.way)
        # print('shot:', self.args.shot)
        k = self.args.way * self.args.shot

        self.deepemd_layer.mode = 'encoder'
        data = self.deepemd_layer(data)

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

        loss = F.cross_entropy(logits, label)

        acc = accuracy(logits, label)

        # print('acc:', acc)

        return logits, acc, loss

    def set_forward(self, batch):
        data, global_target = batch
        # print('global_targey:', global_target)

        # 打印data的shape 5*(1+16)张3,84,84 C H W
        # print('data.shape:', data.shape)

        # key: 重新排列数组
        # 重排数据
        # new_data = torch.empty((data.size(0), data.size(
        #     1), data.size(2), data.size(3)), dtype=data.dtype)

        # # 重新排列数据
        # for i in range(self.args.way):
        #     for j in range(self.args.shot + self.args.query):
        #         if j < self.args.shot:
        #             new_index = j * self.args.way + i
        #         else:
        #             new_index = self.args.shot * self.args.way + \
        #                 (j - self.args.shot) * self.args.way + i
        #         new_data[new_index] = data[i *
        #                                    (self.args.shot + self.args.query) + j]

        # data = new_data.to(self.device)

        data = data.to(self.device)

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


def load_model(model, dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']
    # load from a parallel meta-trained model
    # 打印pretrained_dict的keys
    print('pretrained_dict.keys():', pretrained_dict.keys())

    if 'encoder' in list(pretrained_dict.keys())[0]:
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        # load from a pretrained model
        pretrained_dict = {'encoder.' + k: v for k,
                           v in pretrained_dict.items()}
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # update the param in encoder, remain others still
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('load model success')
    return model
