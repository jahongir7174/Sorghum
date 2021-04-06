import math
import random
import shutil

import onnx
import torch
from PIL import Image
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, model_helper
from caffe2.python.onnx.backend import Caffe2Backend
from torchvision.transforms import functional


def torch2onnx(model, shape):
    print("==> Creating PyTorch model")

    inputs = torch.randn(shape, requires_grad=True)
    model(inputs)

    print("==> Exporting model to ONNX format")
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}

    _ = torch.onnx.export(model, inputs, 'weights/model.onnx',
                          export_params=True,
                          verbose=False,
                          input_names=["input0"],
                          output_names=["output0"],
                          keep_initializers_as_inputs=True,
                          dynamic_axes=dynamic_axes,
                          opset_version=10)

    print("==> Loading and checking exported model from ")
    onnx_model = onnx.load('weights/model.onnx')
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Done")


def onnx2caffe():
    print("==> Exporting ONNX to Caffe2 format")
    onnx_model = onnx.load('weights/model.onnx')
    caffe2_init, caffe2_predict = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    caffe2_init_str = caffe2_init.SerializeToString()
    with open('weights/model.init.pb', "wb") as f:
        f.write(caffe2_init_str)
    caffe2_predict_str = caffe2_predict.SerializeToString()
    with open('weights/model.predict.pb', "wb") as f:
        f.write(caffe2_predict_str)
    print("==> Done")


def print_benchmark(shape):
    print("==> Counting FLOPS")
    model = model_helper.ModelHelper(name="model", init_params=False)

    init_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.init.pb', "rb") as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = core.Net(init_net_proto)

    predict_net_proto = caffe2_pb2.NetDef()
    with open('weights/model.predict.pb', "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)

    model.param_init_net.GaussianFill([],
                                      model.net.external_inputs[0].GetUnscopedName(),
                                      shape=shape,
                                      mean=0.0,
                                      std=1.0)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.BenchmarkNet(model.net.Proto().name, 5, 100, True)
    print("==> Done")


def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def save_checkpoint(state, is_best):
    torch.save(state, f'weights/last.pt')
    if is_best:
        shutil.copyfile(f'weights/last.pt', f'weights/best.pt')


def accuracy(output, target, top_k=(1,)):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n=1):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class RandomResize:
    def __init__(self, size=224):
        self.size = size

        self.scale = (0.08, 1.0)
        self.ratio = (3. / 4., 4. / 3.)
        self.interpolation = (Image.BILINEAR, Image.BICUBIC)

    @staticmethod
    def get_params(img, scale, ratio):
        for _ in range(10):
            target_area = random.uniform(*scale) * img.size[0] * img.size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return functional.resized_crop(img, i, j, h, w, [self.size, self.size], interpolation)
