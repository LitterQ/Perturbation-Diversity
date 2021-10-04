import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torch import autograd
import utils
import math

from utils import softCrossEntropy
from utils import one_hot_tensor, label_smoothing
import ot
import pickle
#from models_new.dis import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attack_None(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        return outputs, None


class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach()


class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FeaScatter, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()



        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None
        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size
        #discriminator = Discriminator().cuda()

        #logits = aux_net(inputs)[0]

        logits, test_fea_nat = aux_net(inputs)
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        alpha = torch.rand(x.size(0), 10, 1, 1)

        #logits_pred_nat, fea_nat, _ = aux_net(inputs)
        logits_pred_nat, fea_nat = self.basic_net(inputs)
        #logits_pred_nat_D = logits_pred_nat.expand_as(alpha)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)
        y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        loss_ce = softCrossEntropy()
        criterion_kl = nn.KLDivLoss(size_average=False)

        iter_num = self.num_steps
        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred = aux_net(x)[0]
            #ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,logits_pred, None, None, 0.01, m, n)
            ot_loss = criterion_kl(F.log_softmax(logits_pred, dim=1), F.softmax(logits_pred_nat, dim=1))
            #cla_loss = self.loss_func(logits_pred, y_tensor_adv)
            #cla_loss = cla_loss.mean()
            pert = torch.sign(x - inputs)
            pert = pert.view(batch_size, 3*32*32)

            mat = torch.matmul(pert, pert.t())
            reg = -torch.logdet(mat + 1e-6 * torch.eye(batch_size).cuda()).mean()


            ot_loss = ot_loss - 10 * batch_size * reg
            aux_net.zero_grad()
            adv_loss = ot_loss

            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv, requires_grad=True)


        logits_pred_2, fea = self.basic_net(x)
        #logits_pred_2, fea = self.basic_net(inputs)


        self.basic_net.zero_grad()
        loss_trades = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_pred_2, dim=1),
                                                                    F.softmax(logits_pred_nat, dim=1))

        #y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        #nat_loss = loss_ce(logits_pred_nat, y_sm.detach())
        nat_loss = loss_ce(logits_pred_2, y_sm.detach())
        adv_loss = nat_loss + 6 * loss_trades


        return logits_pred_2, adv_loss

