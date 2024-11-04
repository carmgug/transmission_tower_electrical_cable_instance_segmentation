import torch
from pycocotools.coco import COCO

def convert_coco_poly_to_mask(anns, coco_dataset: COCO):
    masks = []
    for ann in anns:
        mask = coco_dataset.annToMask(ann)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        masks.append(mask)
    # Concatenate the tensor list along the first dimension (0)
    masks = torch.stack(masks, dim=0)
    return masks

class CocoPolyToMaskConverter:
    """
    Converts the coco annotations into tensors
    """

    def __call__(self, id, coco_dataset: COCO):
        # Take the image
        image_data = coco_dataset.loadImgs(id)[0]
        image_name = image_data['file_name']
        image_id = id
        w, h = (image_data["width"], image_data["height"])
        # Take the annotations associated with the image
        anns_ids = coco_dataset.getAnnIds(imgIds=image_id)
        anns = coco_dataset.loadAnns(anns_ids) # Array of the annotations associated with the image
        # Take all the bounding boxes associated with the image
        boxes = [obj["bbox"] for obj in anns]  # Every bbox is in the format [x, y, width, height]
        boxes = [[bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]] for bbox in boxes]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # Now boxes stores all the bounding boxes in the image 'image_id' as tensors
        # Maintain only the valid bounding boxes
        # Take the classes associated with the image
        classes = [(obj["category_id"]) for obj in anns]
        classes = torch.tensor(classes, dtype=torch.int64)
        # Take the masks inside the image
        masks = convert_coco_poly_to_mask(anns, coco_dataset)
        # Take the area
        area = torch.tensor([obj["area"] for obj in anns], dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor(image_id, dtype=torch.int64)
        target["area"] = area
        return target