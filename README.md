[Sorghum 2022](https://www.kaggle.com/competitions/sorghum-id-fgvc-9) Kaggle contest source code

### Install

* `pip install timm`

### Train

* `bash ./main.sh $ --train` for training, `$` is number of GPUs

### Results

|      Model       | LR Schedule | Epochs | Top1 |
|:----------------:|:-----------:|-------:|-----:|
| EfficientNetV2-M |    Step     |    450 | 93.3 |

### Reference

* https://github.com/rwightman/pytorch-image-models