from tqdm import tqdm
import torch
import numpy as np
import pycocotools
from pycocotools.cocoeval import COCOeval
import json

class Tester:

    def __init__(self, model, test_loader, device, coco_test, working_dir):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.coco_test = coco_test
        self.working_dir = working_dir

    def update_stats(self, test_values: dict, curr_mAPbbox, curr_mAPsegm, epoch):
        test_values["epoch"].append(epoch)
        test_values["mAPbbox"].append(curr_mAPbbox)
        test_values["mAPsegm"].append(curr_mAPsegm)

    def evaluation(self, epoch, annType="bbox"):
        detections = self.coco_test.loadRes(f'{self.working_dir}/prediction_{epoch}_test.json')
        cocoEval = COCOeval(self.coco_test, detections, annType)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval

    def test_and_eval_one_epoch(self, epoch, threshold=0):
        results = []
        self.model.eval()  # Turns off different settings in the model not needed for evaluation/testing
        progress_bar = tqdm(total=len(self.test_loader), desc="Test")  # Initialize a progress bar
        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.test_loader):
                images: torch.Tensor = batch[0]
                targets: list = batch[1]
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(images)
                curr_target = 0
                for output in outputs: #output is Dict[Tensor], outputs is List[Dict[Tensor]]
                    # Now we need to extract the bounding boxes and masks
                    classes = output["labels"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    boxes = output["boxes"].view(-1, 4).cpu().numpy()
                    masks = output["masks"]
                    masks = masks.view(-1, 700, 700).cpu().numpy()
                    for i in range(masks.shape[0]):
                        image_id = targets[curr_target]["image_id"]  # Tensor that has the image_id inside
                        # The bboxes of the mask r-cnn are in the format (xmin, ymin, xmax, ymax)
                        bbox = boxes[i]  # Bounding box associated to the current mask
                        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        bbox = [round(float(x) * 10) / 10 for x in bbox]
                        maskT = masks[i]
                        maskT[maskT > 0.5] = 1
                        maskT[maskT <= 0.5] = 0
                        rle = pycocotools.mask.encode(np.asfortranarray(maskT.astype(np.uint8)))
                        rle['counts'] = rle['counts'].decode('ascii')
                        results.append({
                            'image_id': int(image_id),
                            'category_id': int(classes[i]),
                            'bbox': bbox,
                            'segmentation': rle,
                            'score': float(scores[i])
                        })
                    curr_target += 1
                progress_bar_dict = dict(curr_epoch=epoch, curr_batch=batch_idx)
                progress_bar.set_postfix(progress_bar_dict)
                progress_bar.update()
        # Save file
        with open(f'{self.working_dir}/prediction_{epoch}_test.json', 'w') as fp:
            json.dump(results, fp)
        curr_mAPbbox = round(self.evaluation(epoch, annType="bbox").stats[1], 3)  # Decimal truncating
        curr_mAPsegm = round(self.evaluation(epoch, annType="segm").stats[1], 3)  # Decimal truncating
        progress_bar.close()
        return curr_mAPbbox, curr_mAPsegm