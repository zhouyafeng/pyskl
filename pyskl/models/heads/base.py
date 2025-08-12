# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from ...core import top_k_accuracy
from ..builder import build_loss

from ..losses.class_specific_contrastive_loss import Class_Specific_Contrastive_Loss


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 joint_cfg=None,  # reference 'protogcn',
                 weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.weight = weight

        if joint_cfg is not None:
            if joint_cfg == 'nturgb+d':  # 25*25=625
                n_channel = 625
            elif joint_cfg == 'coco_new':   # 20*20=400
                n_channel = 400
            else:
                raise ValueError(f'Unsupported joint_cfg: {joint_cfg}')
            self.csc_loss = Class_Specific_Contrastive_Loss(num_classes, n_channel)

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, get_graph, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        # loss_cls may be dictionary or single tensor
        loss_cls = self.loss_cls(cls_score, label, **kwargs)
        if get_graph is not None:
            loss_csc = self.csc_loss(get_graph, label.detach(), cls_score.detach())
            if isinstance(loss_cls, dict):
                loss_cls = sum(loss_cls.values())
            loss_cls = loss_cls.mean() + self.weight * loss_csc.mean()
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
