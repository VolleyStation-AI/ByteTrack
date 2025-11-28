#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import contextlib
import io
import itertools
import json
import tempfile
import time


import re
import numpy as np
from pycocotools.cocoeval import COCOeval

class CustomCOCOevalAlmost(COCOeval):
    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        iouType="segm",
        *,  # everything after this must be keyword-only
        custom_bins=None,        # e.g. [[0, 16**2], [16**2, 32**2], ...]
        custom_labels=None,      # e.g. ["xs", "s", "m", "l", "xl"]
        filename_regex=None      # e.g. r"(set1|set3)"
    ):
        super().__init__(cocoGt, cocoDt, iouType)

        # Optional: filter images upfront via regex on file_name
        self.filename_regex = filename_regex
        if self.filename_regex:
            rgx = re.compile(self.filename_regex)
            keep_img_ids = [
                img["id"]
                for img in self.cocoGt.dataset.get("images", [])
                if rgx.search(img.get("file_name", ""))
            ]
            # If regex yields nothing, keep current imgIds to avoid empty eval
            if keep_img_ids:
                self.params.imgIds = keep_img_ids

        # Optional: replace area bins + labels
        if custom_bins is not None and custom_labels is not None:
            assert len(custom_bins) == len(custom_labels), \
                "custom_bins and custom_labels must be same length"
            # You can include an 'all' bin yourself if you want it reported.
            self.params.areaRng = custom_bins
            self.params.areaRngLbl = custom_labels


class CustomCOCOeval(COCOeval):
    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        iouType="segm",
        *,  # extras are keyword-only
        custom_bins=None,        # list of [min_area, max_area]
        custom_labels=None,      # list of labels matching bins
        filename_regex=None,     # regex to filter images by file_name
        ensure_all_bin=True,     # auto-prepend 'all' if missing
        all_bin=(0.0, 1e10),     # default 'all' coverage (pixel^2)
    ):
        super().__init__(cocoGt, cocoDt, iouType)
        self.filename_regex = filename_regex

        # Filter images by filename regex (optional)
        if self.filename_regex:
            rgx = re.compile(self.filename_regex)
            keep_img_ids = [
                img["id"]
                for img in self.cocoGt.dataset.get("images", [])
                if rgx.search(img.get("file_name", ""))
            ]
            if keep_img_ids:
                self.params.imgIds = keep_img_ids

        # Install custom bins/labels
        if custom_bins is not None and custom_labels is not None:
            assert len(custom_bins) == len(custom_labels), "bins and labels must match"
            bins = list(map(list, custom_bins))
            labels = list(custom_labels)

            # Make sure 'all' exists for compatibility with stock summarize() calls
            if ensure_all_bin and ("all" not in labels):
                bins.insert(0, [all_bin[0], all_bin[1]])
                labels.insert(0, "all")

            self.params.areaRng = bins
            self.params.areaRngLbl = labels

    # A summarize that iterates over whatever labels we have
    def summarize_custom(self, dump_json_path=None, maxDets=None):
        """
        Prints AP@[.50:.95], AP50, AP75 and AR@[.50:.95] for each area label present
        in self.params.areaRngLbl. Optionally writes a JSON with the same info.
        Call after self.evaluate(); self.accumulate().
        """
        if not hasattr(self, 'eval') or self.eval is None:
            raise RuntimeError("Call evaluate() and accumulate() before summarize_custom().")

        p = self.params
        eval = self.eval
        T = len(p.iouThrs)
        area_labels = p.areaRngLbl
        maxDets = p.maxDets[-1] if maxDets is None else maxDets

        results = {}

        def mean_ignore_neg1(x):
            x = x[x > -1]
            return -1.0 if x.size == 0 else float(np.mean(x))

        # precision: [TxRxKxAxM], recall: [TxKxAxM]
        precision = eval['precision']
        recall = eval['recall']

        for a_lbl in area_labels:
            # find indices
            aind = [i for i, a in enumerate(p.areaRngLbl) if a == a_lbl]
            mind = [i for i, m in enumerate(p.maxDets) if m == maxDets]
            if not aind or not mind:
                continue

            # AP@[.50:.95] over all IoUs
            ps = precision[:, :, :, aind, mind]  # T x R x K x 1 x 1
            ap_all = mean_ignore_neg1(ps)

            # AP50 and AP75 (if present)
            def ap_at_iou(iou):
                if iou in p.iouThrs:
                    t = np.where(np.isclose(p.iouThrs, iou))[0][0]
                    return mean_ignore_neg1(precision[t, :, :, aind, mind])
                return -1.0

            ap50 = ap_at_iou(0.50)
            ap75 = ap_at_iou(0.75)

            # AR@[.50:.95]
            rs = recall[:, :, aind, mind]  # T x K x 1 x 1
            ar_all = mean_ignore_neg1(rs)

            results[a_lbl] = {
                "AP@[.50:.95]": ap_all,
                "AP50": ap50,
                "AP75": ap75,
                "AR@[.50:.95]": ar_all,
                "maxDets": maxDets
            }

            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area={a_lbl:>7} | maxDets={maxDets:3d} ] = {ap_all:0.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50      | area={a_lbl:>7} | maxDets={maxDets:3d} ] = {ap50:0.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.75      | area={a_lbl:>7} | maxDets={maxDets:3d} ] = {ap75:0.3f}")
            print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area={a_lbl:>7} | maxDets={maxDets:3d} ] = {ar_all:0.3f}")

        if dump_json_path:
            with open(dump_json_path, "w") as f:
                json.dump(results, f, indent=2)
        return results


class COCOEvaluatorCustom:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            alist = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(alist)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                try:
                    label = self.dataloader.dataset.class_ids[int(cls[ind])]
                except:
                    breakpoint()
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            # from yolox.layers import COCOeval_opt as COCOeval

            custom_bins = [
                [16**2, 32**2],      # S
                [32**2, 64**2],      # M
                [64**2, 128**2],     # L
                [128**2, 1e5**2],    # XL
            ]
            custom_labels = ['small', 'medium', 'large', 'xlarge']

            subsets = {
                'texas': r'(texas)',
                'villanova': r'(villanova)',
                'kentucky': r'(kentucky)',
                'arizonastate': r'(arizonastate)',
            }
            for subset, subset_regex in subsets.items():
                logger.info(f"Evaluate COCO stats for subset {subset}")
                
                cocoEval = CustomCOCOeval(
                    cocoGt, cocoDt,
                    iouType=annType[1],
                    custom_bins=custom_bins,
                    custom_labels=custom_labels,
                    filename_regex=subset_regex,  # r'(washington_colorado|arizonastate)'
                )
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()

                logger.info('\n' + redirect_string.getvalue() + '\n')

            # Do normal eval
            cocoEval = CustomCOCOeval(
                cocoGt, cocoDt,
                iouType=annType[1],
                custom_bins=custom_bins,
                custom_labels=custom_labels
            )
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()

            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info



