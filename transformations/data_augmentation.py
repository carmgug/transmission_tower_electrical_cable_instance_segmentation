import torch
import torchvision.transforms.v2 as v2

def get_transforms(train=True):
    if train:
        return v2.Compose([
            v2.RandomChoice([
                v2.RandomHorizontalFlip(p=1),
                v2.RandomVerticalFlip(p=1),
                v2.RandomRotation([-90, 90]),
                v2.RandomResizedCrop(size=[700, 700], scale=(0.1, 2.0), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
            ]),
            v2.RandomApply(torch.nn.ModuleList([
                v2.ColorJitter(
                    brightness=(0.875, 1.125),
                    contrast=(0.5, 1.5),
                    saturation=(0.5, 1.5),
                    hue=(-0.05, 0.05)
                )
            ]), p=0.4),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes()
        ])
    else:
        return v2.Compose([])