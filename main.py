import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from torchinfo import summary
import utils
from tester import Tester
from trainer import Trainer
from utils import pil2tensor
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from custom_coco_dataset import CustomCocoDataset
import time
import os
import datetime

# Training, test and log paths
IMGSRC_TRAINING = "./trainingset"
IMGSRC_TEST= "./testset"

# Setup hyperparameters
START_EPOCH = 1
NUM_EPOCHS = 250
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# Setup checkpoint and working directory
CHECKPOINT_DIR = "./checkpoints"
WORKING_DIR = "./working"

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the JSON file with the information on the images, categories and annotations
json_file_path_tr = utils.myResourcePath("train.json", IMGSRC_TRAINING)
json_file_path_test=utils.myResourcePath("test.json", IMGSRC_TEST)

#Carichiamo il file COCO
coco_tr = COCO(json_file_path_tr)
coco_test = COCO(json_file_path_test)

# Show the dataset info
print("*"*50)
print("Training set info")
print("*"*50)
utils.printDatasetInfo(coco_tr)
print("*"*50)
print("Test set info")
print("*"*50)
utils.printDatasetInfo(coco_test)

# Variable to draw the bounding boxes
cmap = plt.get_cmap('tab20b')
categories = coco_tr.loadCats(coco_tr.getCatIds())
COLORS = [cmap(i) for i in np.linspace(0, 1, len(categories))]
coco: COCO = coco_tr

# Load the dataset
print("Creating data loaders")
train_dataset = CustomCocoDataset(IMGSRC_TRAINING, json_file_path_tr, pil2tensor, aug=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: (torch.stack([item[0] for item in x],dim=0), [item[1] for item in x])) #x = [(img1, img2, img3,...), ()]
test_dataset=CustomCocoDataset(IMGSRC_TEST, json_file_path_test, pil2tensor, aug=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=lambda x: (torch.stack([item[0] for item in x],dim=0), [item[1] for item in x])) #x = [(img1, img2, img3,...), ()]

# Load the model
print("Loading Mask R-CNN model")
torch.cuda.empty_cache()
num_of_classes=len(coco.getCatIds())+1 # Including the background (5)
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_of_classes)
# Now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# Replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_of_classes)
model=model.to(device)
print(model)

# Summary of the model
summary(model,
        input_size=[7,3,700,700],
        col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])

# Set other parameters
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
epoch_count = []
train_loss_values = {
    "avg_loss": []
}
best_mAPbbox = 0.0
best_mAPsegm = 0.0
test_values = {
    "epoch": [],
    "mAPbbox": [],
    "mAPsegm": []
}
lr_scheduler = None

# Initialize trainer and tester objects
trainer = Trainer(model, train_loader, optimizer, lr_scheduler, device, LEARNING_RATE)
tester = Tester(model, test_loader, device, coco_test, WORKING_DIR)

# RESUME THE TRAINING FROM A CHECKPOINT
RESUME = False
if RESUME:
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "checkpoint.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    START_EPOCH = checkpoint["epoch"] + 1
    epoch_count = checkpoint["list_epoch"]
    train_loss_values= checkpoint["dict_losses_tr"]
    best_mAPbbox = checkpoint["best_mAPbbox"]
    best_mAPsegm = checkpoint["best_mAPsegm"]

# START TRAINING THE MODEL

print("Start training")
start_time = time.time()
for epoch in range(START_EPOCH, NUM_EPOCHS+1):
        avg_loss, epoch_loss_model, lr_scheduler = trainer.train_one_epoch(epoch)
        epoch_count.append(epoch)
        # Save the average loss for this epoch
        train_loss_values["avg_loss"].append(avg_loss)
        for key, value in epoch_loss_model.items():
                if key not in train_loss_values:
                        train_loss_values[key] = []
                train_loss_values[key].append(value / len(train_loader))
        # Save a checkpoint
        checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "list_epoch": epoch_count,
                "dict_losses_tr": train_loss_values,
                "dict_stats_test": test_values,
                "best_mAPbbox": best_mAPbbox,
                "best_mAPsegm": best_mAPsegm
        }
        utils.save_on_master(checkpoint, os.path.join(CHECKPOINT_DIR, "checkpoint.pth"))
        print("Checkpoint saved.")
        if epoch % 10 == 0 or epoch > 50: # Start evaluating the model every 10 epoch, and then at every epoch starting from the 50th
                print(f"Epoch Test: {epoch}")
                curr_mAPbbox, curr_mAPsegm = tester.test_and_eval_one_epoch(model, epoch)
                tester.update_stats(test_values, curr_mAPbbox, curr_mAPsegm, epoch)
                if (curr_mAPsegm > best_mAPsegm) or (best_mAPsegm == curr_mAPsegm and curr_mAPbbox > best_mAPbbox):
                        # If there's an improvement on the segmentation or the bboxes, then save the model
                        best_mAPsegm = curr_mAPsegm
                        best_mAPbbox = curr_mAPbbox
                        checkpoint = {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "list_epoch": epoch_count,
                                "dict_losses_tr": train_loss_values,
                                "dict_stats_test": test_values,
                                "best_mAPbbox": best_mAPbbox,
                                "best_mAPsegm": best_mAPsegm
                        }
                        utils.save_on_master(checkpoint, os.path.join(CHECKPOINT_DIR, "checkpoint_best.pth"))
                        print("Best model saved.")
                else:
                        print("No improvement of the mAP values. No best model saved.")
        # Checkpoint to save test_values
        checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "list_epoch": epoch_count,
                "dict_losses_tr": train_loss_values,
                "dict_stats_test": test_values,
                "best_mAPbbox": best_mAPbbox,
                "best_mAPsegm": best_mAPsegm
        }
        utils.save_on_master(checkpoint, os.path.join(CHECKPOINT_DIR, "checkpoint.pth"))
        # Apply the copy-paste from the 200th epoch on the not augmented training set
        if epoch == 200:
                train_dataset = CustomCocoDataset(IMGSRC_TRAINING, json_file_path_tr, pil2tensor, aug=False)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: (torch.stack([item[0] for item in x],dim=0), [item[1] for item in x])) #x = [(img1, img2, img3,...), ()]

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f"Training time: {total_time_str}")