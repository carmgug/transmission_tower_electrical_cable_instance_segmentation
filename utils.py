import os
from pycocotools.coco import COCO
import pycocotools.mask as mask
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2

pil2tensor = ToTensor()
tensor2pil = ToPILImage()

def myResourcePath(fname, directory):
    filename = os.path.join(directory, fname)
    if not os.path.exists(filename):
        raise RuntimeError(f'file not found {filename}')
    return filename

def printDatasetInfo(coco_obj: COCO):
    """
    Given a coco object, returns the dataset info
    """
    # Retrieve categories
    categories = coco_obj.loadCats(coco_obj.getCatIds())
    # Print ID and name of categories
    print("Available categories:")
    for category in categories:
        print(f"   ID: {category['id']}, Name: {category['name']}")
    # Print the total number of images in the dataset
    total_images = len(coco_obj.getImgIds())
    print(f"\nTotal number of images in the dataset: {total_images}")
    # Print the number of images and instances for each category
    print("\nDetails per category:")
    for category in categories:
        cat_id = category['id']
        curr_name = category['name']
        image_ids = coco_obj.getImgIds(catIds=cat_id)  # Images that contain the category 'cat_id'
        instance_ids = coco_obj.getAnnIds(catIds=cat_id)  # Annotations of the category 'cat_id'
        print(f"\nCategory: {curr_name}")
        print(f"   Number of images: {len(image_ids)}")
        print(f"   Number of instances: {len(instance_ids)}")

def getImgByID2Tensor(img_id, coco_dataset: COCO, source):
    image_data = coco_dataset.loadImgs(img_id)[0]
    # Load the original image
    image_path = myResourcePath(image_data['file_name'], source)
    image = Image.open(image_path)
    # Take the image and transform it into a tensor
    image = pil2tensor(image)
    return image

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

# DISPLAY IMAGES

def print_legend(colors, labels, ax):
    pat = [patches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=pat, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=len(colors))

def dislplayImageWithBoxesAndMasksWithRle(coco, colors, categories, IMGSRC, image_id, display_type="both"):
    """
    Given an image, returns the associated mask.
    """
    # Load the image data
    image_data = coco.loadImgs(image_id)[0]
    # Load the original image
    image_path = myResourcePath(image_data['file_name'], IMGSRC)
    image = np.array(Image.open(image_path))
    # Load the image annotations
    anns_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    anns = coco.loadAnns(anns_ids)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, interpolation='nearest')
    axs[0].set_title("Original image")
    # Display bounding boxes
    if display_type in ['box', 'both']:
        for ann in anns:
            box = ann['bbox']
            category_id = ann["category_id"]
            bb = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=colors[category_id - 1], facecolor="none")
            axs[1].add_patch(bb)
        axs[1].imshow(image)
        print_legend(colors, [cat_name['name'] for cat_name in categories], axs[1])
        axs[1].set_title("BBOX")
    # Display segmentation polygon
    if display_type in ['seg', 'both']:
        for ann in anns:
            category_id = ann["category_id"]
            for seg in ann['segmentation']:
                poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                polygon = patches.Polygon(poly, closed=False, edgecolor=colors[category_id - 1], fill=False)
                axs[2].add_patch(polygon)
        axs[2].imshow(image)
        axs[2].set_title("segmentation polygon")
    plt.show()

def polygonFromMask(maskedArr):
    """
    Taken from: https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    """
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    if len(segmentation)==0:
      return segmentation
    RLEs = mask.frPyObjects(segmentation,maskedArr.shape[0],maskedArr.shape[1])
    RLE = mask.merge(RLEs)
    area = mask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)
    return segmentation[0] #, [x, y, w, h], area

def displayImageWithBoxesAndMasksWithCounts(coco, COLORS, image_id, IMGSRC, categories, display_type="both"):
    """
    Given an image, returns the associated mask starting from counts
    """
    # Load the image data
    image_data = coco.loadImgs(image_id)[0]
    # Load the original image
    image_path = myResourcePath(image_data['file_name'], IMGSRC)
    image = np.array(Image.open(image_path))
    # Create the plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Visualize the original image
    axs[0].imshow(image, interpolation='nearest')
    axs[0].set_title("Original image")
    # Load the image annotations
    anns_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    anns = coco.loadAnns(anns_ids)
    # Display the bounding boxes
    if display_type in ['box', 'both']:
        for ann in anns:
            if (ann["score"] > 0):
                box = ann['bbox']
                category_id = ann["category_id"] - 1
                bb = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=COLORS[category_id], facecolor="none")
                axs[1].add_patch(bb)
        axs[1].imshow(image)
        print_legend(COLORS, [cat_name['name'] for cat_name in categories], axs[1])
        axs[1].set_title("BBOX")
    # Display segmentation polygon
    if display_type in ['seg', 'both']:
        for ann in anns:
            if (ann["score"] > 0):
                category_id = ann["category_id"] - 1
                seg = ann["segmentation"]
                coco_counts_data = seg["counts"]
                maskedArr = mask.decode(seg)
                polygon = polygonFromMask(maskedArr)
                if len(polygon) != 0:
                    poly = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                    polygon = patches.Polygon(poly, closed=False, edgecolor=COLORS[category_id], fill=False)
                    axs[2].add_patch(polygon)
        axs[2].imshow(image)
        axs[2].set_title("segmentation polygon")
    plt.show()

def displayImageWithBoxesAndMaskArray(tensor_image: torch.Tensor, target, COLORS, categories, display_type="both"):
    """
    Given an image, returns the associated mask starting from counts.
    """
    img = tensor2pil(tensor_image)
    classes = target["labels"].cpu().numpy()
    boxes = target["boxes"].cpu().numpy()
    masksArr = target["masks"].numpy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img, interpolation='nearest')
    axs[0].set_title("Original image")
    # Display the bounding boxes
    if display_type in ['box', 'both']:
        index = 0
        for bbox in boxes:
            category_id = classes[index] - 1
            bb = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor=COLORS[category_id], facecolor="none")
            axs[1].add_patch(bb)
            index += 1
        axs[1].imshow(img)
        print_legend(COLORS, [cat_name['name'] for cat_name in categories], axs[1])
        axs[1].set_title("BBOX")
    # Display segmentation polygon
    if display_type in ['seg', 'both']:
        index = 0
        for maskArr in masksArr:
            category_id = classes[index] - 1
            polygon = polygonFromMask(maskArr)
            if len(polygon) != 0:
                poly = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                polygon = patches.Polygon(poly, closed=False, edgecolor=COLORS[category_id], fill=False)
                axs[2].add_patch(polygon)
            index += 1
        axs[2].imshow(img)
        axs[2].set_title("segmentation polygon")
    plt.show()