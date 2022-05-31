import argparse
import copy
import csv
import os
import warnings

import numpy
import torch
import tqdm
from timm import utils
from torch.utils import data
from torchvision import transforms

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")

data_dir = os.path.join('..', 'Dataset', 'Sorghum')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def lr(args):
    return 0.001 * args.batch_size * args.world_size


def cut_mix(samples, targets, model, criterion):
    shape = samples.size()
    index = torch.randperm(shape[0]).cuda()
    alpha = numpy.sqrt(1. - numpy.random.beta(1.0, 1.0))

    w = numpy.int(shape[2] * alpha)
    h = numpy.int(shape[3] * alpha)

    # uniform
    c_x = numpy.random.randint(shape[2])
    c_y = numpy.random.randint(shape[3])

    x1 = numpy.clip(c_x - w // 2, 0, shape[2])
    y1 = numpy.clip(c_y - h // 2, 0, shape[3])
    x2 = numpy.clip(c_x + w // 2, 0, shape[2])
    y2 = numpy.clip(c_y + h // 2, 0, shape[3])

    samples[:, :, x1:x2, y1:y2] = samples[index, :, x1:x2, y1:y2]

    alpha = 1 - ((x2 - x1) * (y2 - y1) / (shape[-1] * shape[-2]))

    samples = samples.cuda()
    targets = targets.cuda()

    with torch.cuda.amp.autocast():
        outputs = model(samples)
    return criterion(outputs, targets) * alpha + criterion(outputs, targets[index]) * (1. - alpha)


def train(args):
    model = nn.create()
    ema_m = nn.EMA(model)

    amp_scale = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.SGD(util.weight_decay(model), lr(args), 0.9, nesterov=True)

    if not args.distributed:
        model = torch.nn.parallel.DataParallel(model)
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank])

    criterion = nn.PolyLoss().cuda()
    scheduler = nn.CosineLR(optimizer, args)

    with open(f'./weights/step.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
            writer.writeheader()

        sampler = None
        dataset = Dataset(os.path.join(data_dir, 'train'),
                          transforms.Compose([util.Resize(args.input_size),
                                              transforms.RandomHorizontalFlip(),
                                              util.MixAugment(), util.RandomAugment(),
                                              transforms.ToTensor(), normalize, util.Cutout()]))

        if args.distributed:
            sampler = data.distributed.DistributedSampler(dataset)

        loader = data.DataLoader(dataset, args.batch_size, not args.distributed,
                                 sampler=sampler, num_workers=8, pin_memory=True)

        model.train()
        for epoch in range(args.epochs):
            m_loss = util.AverageMeter()

            if args.distributed:
                sampler.set_epoch(epoch)
            p_bar = loader
            if args.local_rank == 0:
                print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
                p_bar = tqdm.tqdm(p_bar, total=len(loader))

            optimizer.zero_grad()

            for samples, targets in p_bar:
                loss = cut_mix(samples, targets, model, criterion)

                optimizer.zero_grad()

                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                ema_m.update(model)

                if args.distributed:
                    utils.distribute_bn(model, args.world_size, True)
                    loss = utils.reduce_tensor(loss.data, args.world_size)

                m_loss.update(loss.item(), samples.size(0))

                if args.local_rank == 0:
                    gpus = '%.4gG' % (torch.cuda.memory_reserved() / 1E9)
                    desc = ('%10s' * 2 + '%10.3g') % ('%g/%g' % (epoch + 1, args.epochs), gpus, m_loss.avg)

                    p_bar.set_description(desc)

            scheduler.step(epoch + 1)

            if args.local_rank == 0:
                writer.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'loss': str(f'{m_loss.avg:.3f}')})

                state = {'model': copy.deepcopy(ema_m.model).half()}
                torch.save(state, './weights/model.pt')

                del state

        del loader
        del sampler
        del dataset

    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(args):
    from utils.dataset import TestDataset
    model = torch.load('weights/model.pt', 'cuda')['model'].float()
    model.half()
    model.eval()

    dataset = TestDataset(data_dir,
                          transforms.Compose([transforms.Resize(size=args.input_size),
                                              transforms.CenterCrop(args.input_size),
                                              transforms.ToTensor(), normalize]))
    loader = data.DataLoader(dataset, 32, num_workers=4)

    with open(f'./weights/submission.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'cultivar'])
        writer.writeheader()
        for samples, filenames in tqdm.tqdm(loader):
            samples = samples.cuda()
            samples = samples.half()

            with torch.no_grad():
                for filename, output in zip(filenames, model(samples)):
                    output = output.argmax()
                    output = int(output.cpu().numpy())
                    output = dataset.idx_to_cls[output]

                    writer.writerow({'filename': filename, 'cultivar': output})


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=450, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    utils.random_seed(0, args.local_rank)

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
