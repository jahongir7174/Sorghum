import copy
import math

import torch


def round_filters(filters, width, divisor=8):
    filters *= width
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def activation_fn():
    return torch.nn.Hardswish(inplace=True)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.uniform_(-1.0 / math.sqrt(m.weight.size()[0]), 1.0 / math.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, k // 2, 1, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class SE(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // 4, 1),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Conv2d(ch // 4, ch, 1),
                                      torch.nn.Hardsigmoid(inplace=True))

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))


class Residual(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, mid_ch, out_ch, activation, k, s, se):
        super().__init__()
        identity = torch.nn.Identity
        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(Conv(in_ch, mid_ch, activation) if in_ch != mid_ch else identity(),
                                       Conv(mid_ch, mid_ch, activation, k, s, mid_ch),
                                       SE(mid_ch) if se else identity(),
                                       Conv(mid_ch, out_ch, identity()))

    def forward(self, x):
        return self.res(x) + x if self.add else self.res(x)


class MobileNetV3(torch.nn.Module):
    def __init__(self, w) -> None:
        super().__init__()
        out_filters = [16, 24, 40, 80, 112, 160, 960, 1280]
        mid_filters = [16, 64, 72, 120, 240, 200, 184, 480, 672, 960]

        out_filters = [round_filters(width, w) for width in out_filters]
        mid_filters = [round_filters(width, w) for width in mid_filters]

        relu_fn = torch.nn.ReLU(inplace=True)
        feature = [Conv(3, out_filters[0], activation_fn(), 3, 2),
                   Residual(out_filters[0], mid_filters[0], out_filters[0], relu_fn, 3, 1, False),
                   Residual(out_filters[0], mid_filters[1], out_filters[1], relu_fn, 3, 2, False),
                   Residual(out_filters[1], mid_filters[2], out_filters[1], relu_fn, 3, 1, False),
                   Residual(out_filters[1], mid_filters[2], out_filters[2], relu_fn, 5, 2, True),
                   Residual(out_filters[2], mid_filters[3], out_filters[2], relu_fn, 5, 1, True),
                   Residual(out_filters[2], mid_filters[3], out_filters[2], relu_fn, 5, 1, True),
                   Residual(out_filters[2], mid_filters[4], out_filters[3], activation_fn(), 3, 2, False),
                   Residual(out_filters[3], mid_filters[5], out_filters[3], activation_fn(), 3, 1, False),
                   Residual(out_filters[3], mid_filters[6], out_filters[3], activation_fn(), 3, 1, False),
                   Residual(out_filters[3], mid_filters[6], out_filters[3], activation_fn(), 3, 1, False),
                   Residual(out_filters[3], mid_filters[7], out_filters[4], activation_fn(), 3, 1, True),
                   Residual(out_filters[4], mid_filters[8], out_filters[4], activation_fn(), 3, 1, True),
                   Residual(out_filters[4], mid_filters[8], out_filters[5], activation_fn(), 5, 2, True),
                   Residual(out_filters[5], mid_filters[9], out_filters[5], activation_fn(), 5, 1, True),
                   Residual(out_filters[5], mid_filters[9], out_filters[5], activation_fn(), 5, 1, True),
                   Conv(out_filters[5], out_filters[6], activation_fn())]

        self.feature = torch.nn.Sequential(*feature)
        self.fc = torch.nn.Sequential(torch.nn.Conv2d(out_filters[6], out_filters[7], 1),
                                      activation_fn(),
                                      torch.nn.Dropout(0.2, True),
                                      torch.nn.Conv2d(out_filters[7], 1000, 1),
                                      torch.nn.Flatten())

        initialize_weights(self)

    def forward(self, x):
        x = self.feature(x)
        return self.fc(x.mean((2, 3), keepdim=True))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                delattr(m, 'norm')
                m.forward = m.fuse_forward
        return self


class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model.module if type(model) is torch.nn.DataParallel else model)
        self.model.eval()

    def update_fn(self, model, fn):
        with torch.no_grad():
            model = model.module if type(model) is torch.nn.DataParallel else model
            for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(fn(ema_v, model_v))

    def update(self, model):
        self.update_fn(model, fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


class CrossEntropyLoss(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1512.00567.pdf]
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        prob = self.softmax(x)
        loss = -prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        return ((1. - self.epsilon) * loss + self.epsilon * (-prob.mean(dim=-1))).mean()


class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        for param_group in self.optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        self.base_values = [param_group['initial_lr'] for param_group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.base_values]
        self.update_groups(self.warmup_lr_init)

    def __str__(self) -> str:
        return 'step'

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.base_values]
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class RMSprop(torch.optim.Optimizer):
    """
    [https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf]
    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0.,
                 centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss
