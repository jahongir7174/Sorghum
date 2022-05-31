import math

import torch
from torch.nn.functional import one_hot


def create():
    from timm import create_model

    model = create_model(model_name='tf_efficientnetv2_m', pretrained=True,
                         num_classes=100, drop_rate=0.4, drop_path_rate=0.2)

    model.cuda()

    return model


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class Tune(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        in_channels = 0
        self.models = torch.nn.ModuleList()

        for weight in weights:
            model = torch.load(weight, 'cuda')['model'].float()

            model.eval()

            for param in model.parameters():
                param.requires_grad_(False)

            self.models.append(model)

            in_channels += model(torch.randn(2, 3, 224, 224).cuda()).shape[1]
        self.fc = torch.nn.Sequential(Conv(in_channels, 1280, torch.nn.SiLU()),
                                      torch.nn.AdaptiveAvgPool2d(output_size=1),
                                      torch.nn.Flatten(), torch.nn.Linear(1280, 100))
        for m in self.fc.modules():
            if isinstance(m, torch.nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                torch.nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, torch.nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.size()[0])
                torch.nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        y = []
        for model in self.models:
            y.append(model(x))
        y = torch.cat(y, dim=-1)
        return self.fc(y)


class EMA:
    def __init__(self, model):
        super().__init__()
        from copy import deepcopy

        self.model = deepcopy(model).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.module.state_dict().values()

            for e, m in zip(e_std, m_std):
                e.copy_(0.9999 * e + (1. - 0.9999) * m)


class CosineLR:
    def __init__(self, optimizer, args):
        self.optimizer = optimizer

        self.epochs = args.epochs
        self.values = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.warmup_epochs = 5
        self.warmup_values = [(v - 1e-4) / self.warmup_epochs for v in self.values]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1e-4

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [1e-4 + epoch * value for value in self.warmup_values]
        else:
            if epoch < self.epochs:
                alpha = math.pi * (epoch - (self.epochs * (epoch // self.epochs))) / self.epochs
                values = [1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(alpha)) for lr in self.values]
            else:
                values = [1e-5 for _ in self.values]

        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        prob = self.softmax(outputs)
        loss = -(prob.gather(dim=-1, index=targets.unsqueeze(1))).squeeze(1)

        return ((1.0 - self.epsilon) * loss - self.epsilon * prob.mean(dim=-1)).mean()


class PolyLoss(torch.nn.Module):
    def __init__(self, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.Softmax(1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        ce = self.criterion(outputs, targets)
        pt = one_hot(targets, outputs.size()[1]) * self.softmax(outputs)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()
