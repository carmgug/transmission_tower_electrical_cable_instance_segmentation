import torch
from torchvision import tv_tensors
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import utils
from transformations import data_augmentation
from transformations.coco_poly_to_mask_converter import CocoPolyToMaskConverter

class CustomCocoDataset(torch.utils.data.Dataset):
    """
    Creates a custom dataset with coco annotations and converts the annotations into tensors
    """

    def __init__(self, root, ann_file, transform=ToTensor(), aug=False):
        self.coco = COCO(ann_file)
        self.root = root
        self.converter = CocoPolyToMaskConverter()
        self.transform = transform
        self.aug = aug # Perform data augmentation or not

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img = utils.getImgByID2Tensor(img_id, self.coco, self.root)
        # Create the target as a dictionary containing all the necessary information
        target = self.converter(idx, self.coco)
        if self.aug:
            transforms = data_augmentation.get_transforms(train=True)
            img, target = self.applyTransforms(transforms, img, target)
        return img, target

    def applyTransforms(self, transforms, curr_image, curr_target):
        curr_image = tv_tensors.Image(curr_image)
        curr_target['boxes'] = tv_tensors.BoundingBoxes(curr_target['boxes'], format='XYXY', canvas_size=(curr_image.shape[1], curr_image.shape[2]))
        curr_target['masks'] = tv_tensors.Mask(curr_target['masks'], dtype=torch.uint8)
        output = transforms({'image': curr_image, 'boxes': curr_target['boxes'], 'masks': curr_target['masks'], 'labels': curr_target['labels']})
        res_image = output['image']
        res_boxes = output['boxes']
        res_masks = output['masks']
        res_labels = output['labels']
        curr_target['boxes'] = res_boxes
        curr_target['masks'] = res_masks
        curr_target['labels'] = res_labels
        return res_image, curr_target