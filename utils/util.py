import math
import random

import cv2
import numpy
from PIL import Image, ImageOps, ImageEnhance

max_value = 10.


def weight_decay(model):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0}, {'params': p2, 'weight_decay': 1e-5}]


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def rotate(image, magnitude):
    magnitude = (magnitude / max_value) * 30

    if random.random() > 0.5:
        magnitude *= -1

    return image.rotate(magnitude, resample=resample())


def shear_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.3

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), resample=resample())


def shear_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.3

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), resample=resample())


def translate_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.5

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())


def translate_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.5

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[1]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def equalize(image, _):
    return ImageOps.equalize(image)


def invert(image, _):
    return ImageOps.invert(image)


def identity(image, _):
    return image


def normalize(image, _):
    return ImageOps.autocontrast(image)


def brightness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Brightness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Brightness(image).enhance(magnitude)


def color(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Color(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Color(image).enhance(magnitude)


def contrast(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Contrast(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Contrast(image).enhance(magnitude)


def sharpness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Sharpness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Sharpness(image).enhance(magnitude)


def solar(image, magnitude):
    magnitude = int((magnitude / max_value) * 256)
    if random.random() > 0.5:
        return ImageOps.solarize(image, magnitude)
    else:
        return ImageOps.solarize(image, 256 - magnitude)


def poster(image, magnitude):
    magnitude = int((magnitude / max_value) * 4)
    if random.random() > 0.5:
        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)
    else:
        if random.random() > 0.5:
            magnitude = 4 - magnitude
        else:
            magnitude = 4 + magnitude

        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)


def random_hsv(image):
    x = numpy.arange(0, 256, dtype=numpy.int16)
    hsv = numpy.random.uniform(-1, 1, 3) * [.015, .7, .4] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

    lut_hue = ((x * hsv[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * hsv[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * hsv[2], 0, 255).astype('uint8')

    h = cv2.LUT(h, lut_hue)
    s = cv2.LUT(s, lut_sat)
    v = cv2.LUT(v, lut_val)

    image_hsv = cv2.merge((h, s, v)).astype('uint8')
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)


def random_affine(image):
    h = image.shape[0]
    w = image.shape[1]

    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotation = numpy.eye(3)
    a = random.uniform(-30, 30)
    s = random.uniform(1 - 0.25, 1 + 0.25)
    rotation[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-0.1, 0.1) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-0.1, 0.1) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = numpy.eye(3)
    translation[0, 2] = random.uniform(0.5 - 0.2, 0.5 + 0.2) * w  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - 0.2, 0.5 + 0.2) * h  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translation @ shear @ rotation @ perspective @ center
    if (matrix != numpy.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h))  # affine
    return image


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([self.size, self.size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        in_ratio = size[0] / size[1]
        if in_ratio < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w


class Cutout:
    def __init__(self):
        from timm.data.random_erasing import RandomErasing
        self.cutout = RandomErasing(0.25, mode='pixel', device='cpu')

    def __call__(self, image):
        return self.cutout(image)


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        self.num = self.num + n
        self.sum = self.sum + v * n
        self.avg = self.sum / self.num


class MixAugment:
    def __init__(self, mean=4, sigma=0.5, n=4):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, identity, invert, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, sharpness, solar, poster)

    def __call__(self, image):
        aug_image = image.copy()

        for transform in numpy.random.choice(self.transform, self.n):
            magnitude = numpy.random.normal(self.mean, self.sigma)
            magnitude = min(max_value, max(0., magnitude))
            aug_image = transform(aug_image, magnitude)
        alpha = random.random()
        return Image.blend(image, aug_image, alpha if alpha > 0.3 else alpha / 3)


class RandomAffine:
    def __call__(self, image):
        image = numpy.asarray(image)
        random_hsv(image)
        image = random_affine(image)
        return Image.fromarray(image)


class RandomAugment:
    def __init__(self, mean=9, sigma=0.5, n=2):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, identity, invert, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, sharpness, solar, poster)

    def __call__(self, image):
        for transform in numpy.random.choice(self.transform, self.n):
            magnitude = numpy.random.normal(self.mean, self.sigma)
            magnitude = min(max_value, max(0., magnitude))

            image = transform(image, magnitude)
        return image
