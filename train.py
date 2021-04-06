import csv
import os
import random

import numpy as np
import torch
import tqdm
from torch.backends import cudnn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from nets import nn
from utils import util

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

v = 0
versions = ({'w': 1.00, 'r': 224},
            {'w': 1.25, 'r': 224})[v]


def batch_fn(images, target, model, device, loss_fn, training=True):
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    if training:
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(images)
            loss = loss_fn(output, target)
    else:
        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = loss_fn(output, target).data
    acc1, acc5 = util.accuracy(output, target, top_k=(1, 5))
    return loss, acc1, acc5, output


def main():
    epochs = 450
    device = torch.device('cuda')
    data_dir = '../Dataset/IMAGENET'
    num_gpu = torch.cuda.device_count()
    v_batch_size = 16 * num_gpu
    t_batch_size = 256 * num_gpu

    model = nn.MobileNetV3(versions['w']).to(device)
    optimizer = nn.RMSprop(util.add_weight_decay(model), 0.016 * num_gpu, 0.9, 1e-3, momentum=0.9)

    model = torch.nn.DataParallel(model)
    _ = model(torch.zeros(1, 3, versions['r'], versions['r']).to(device))

    ema = nn.EMA(model)
    t_criterion = nn.CrossEntropyLoss(0.1).to(device)
    v_criterion = torch.nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                     transforms.Compose([util.RandomResize(size=versions['r']),
                                                         transforms.ColorJitter(0.4, 0.4, 0.4),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(), normalize]))
    v_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                     transforms.Compose([transforms.Resize(versions['r'] + 32),
                                                         transforms.CenterCrop(versions['r']),
                                                         transforms.ToTensor(), normalize]))

    t_loader = data.DataLoader(t_dataset, batch_size=t_batch_size, shuffle=True,
                               num_workers=os.cpu_count(), pin_memory=True)
    v_loader = data.DataLoader(v_dataset, batch_size=v_batch_size, shuffle=False,
                               num_workers=os.cpu_count(), pin_memory=True)

    scheduler = nn.StepLR(optimizer)
    amp_scale = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    with open(f'weights/{scheduler.__str__()}.csv', 'w') as summary:
        writer = csv.DictWriter(summary, fieldnames=['epoch', 't_loss', 'v_loss', 'acc@1', 'acc@5'])
        writer.writeheader()
        best_acc1 = 0
        for epoch in range(0, epochs):
            print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
            t_bar = tqdm.tqdm(t_loader, total=len(t_loader))
            model.train()
            t_loss = util.AverageMeter()
            v_loss = util.AverageMeter()
            for images, target in t_bar:
                loss, _, _, _ = batch_fn(images, target, model, device, t_criterion)
                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                ema.update(model)
                torch.cuda.synchronize()
                t_loss.update(loss.item(), images.size(0))

                t_bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))
            top1 = util.AverageMeter()
            top5 = util.AverageMeter()

            ema_model = ema.model.eval()
            with torch.no_grad():
                for images, target in tqdm.tqdm(v_loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
                    loss, acc1, acc5, output = batch_fn(images, target, ema_model, device, v_criterion, False)
                    torch.cuda.synchronize()
                    v_loss.update(loss.item(), output.size(0))
                    top1.update(acc1.item(), images.size(0))
                    top5.update(acc5.item(), images.size(0))
                acc1, acc5 = top1.avg, top5.avg
                print('%10.3g' * 2 % (acc1, acc5))

            scheduler.step(epoch + 1)
            writer.writerow({'epoch': epoch + 1,
                             't_loss': t_loss.avg,
                             'v_loss': v_loss.avg,
                             'acc@1': acc1,
                             'acc@5': acc5})
            util.save_checkpoint({'state_dict': ema.model.state_dict()}, acc1 > best_acc1)
            best_acc1 = max(acc1, best_acc1)
    torch.cuda.empty_cache()


def print_parameters():
    model = nn.MobileNetV3(versions['w']).fuse().eval()
    _ = model(torch.zeros(1, 3, versions['r'], versions['r']))
    params = sum(p.numel() for p in model.parameters())
    print('{:<20}  {:<8}'.format('Number of parameters ', int(params)))


def benchmark():
    shape = (1, 3, versions['r'], versions['r'])
    util.torch2onnx(nn.MobileNetV3(versions['w']).fuse().eval(), shape)
    util.onnx2caffe()
    util.print_benchmark(shape)


if __name__ == '__main__':
    if not os.path.exists('weights'):
        os.makedirs('weights')
    print_parameters()
