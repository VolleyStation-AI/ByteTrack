import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from yolox.evaluators.coco_evaluator import COCOEvaluator
from yolox.exp.build import get_exp
from yolox.core import Trainer
from tools.train import make_parser
from loguru import logger

from tqdm import tqdm

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
)

import contextlib
import io
import itertools
import json
import tempfile
import time
import random

class CocoEvalPerClass(COCOEvaluator):
    def __init__(
        self,
        classes: list[str],
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        testdev=False,
    ):
        super().__init__(dataloader, img_size, confthre, nmsthre, num_classes, testdev)
        self.classes = classes

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

        #vis_batches = random.choices(range(len(self.dataloader)), k=20)

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

                """
                if cur_iter in vis_batches:
                    if outputs[0] is not None:
                        save_vis(imgs[0], outputs[0], cur_iter)
                """

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

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
            # from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            results = {}
            all_cat_ids = cocoEval.params.catIds[:]
            for cat_id in all_cat_ids:
                class_name = self.classes[cat_id - 1]
                info += f"Eval for class {cat_id}: {class_name}\n"
                cocoEval.params.catIds = [cat_id]
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                info += redirect_string.getvalue() + "\n"
                results[class_name] = cocoEval.stats[0], cocoEval.stats[1]

            info += "Eval for all classes:\n"
            cocoEval.params.catIds = all_cat_ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue() + "\n"
            results["total"] = cocoEval.stats[0], cocoEval.stats[1]
            return results, info
        else:
            return {}, info


def draw_text(
    img,
    text,
    pos=(0, 0),
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=2,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if text_color_bg is not None:
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, y + text_h + font_scale - 1),
        font,
        font_scale,
        text_color,
        font_thickness,
    )


def save_vis(img, output, i):
    colors = [
        (0x66, 0xCD, 0xAA),
        (0x00, 0xFF, 0x00),
        (0x00, 0x00, 0xFF),
        (0x1E, 0x90, 0xFF),
        (0xFF, 0xA5, 0x00),
        (0xFF, 0xA5, 0x00),
        (0xFF, 0xA5, 0x00),
        (0xFF, 0x14, 0x93),
    ]
    img = img.cpu()
    rgb_means = ((0.485, 0.456, 0.406),)
    std = (0.229, 0.224, 0.225)
    img = img * torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(rgb_means).view(3, 1, 1)

    img = img.permute(1, 2, 0)
    img = (img.numpy() * 255).astype(np.uint8).copy()

    output = output.cpu().round().int()

    for pred in output:
        bbox = pred[:4]
        obj_cls = pred[-1]
        cv2.rectangle(img, bbox[:2].tolist(), (bbox[2:]).tolist(), colors[obj_cls], 2)

    plt.imsave(f"/mnt/h/output/dbg/det_pred/out_{i:04d}.png", img)


def main(exp, args):
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    trainer = Trainer(exp, args)
    # setup model etc
    trainer.before_train()
    testdev = False

    val_loader = exp.get_eval_loader(args.batch_size, trainer.is_distributed, testdev)
    evaluator = CocoEvalPerClass(
        exp.class_names,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        testdev=testdev,
    )

    per_class_eval, summary = evaluator.evaluate(
        trainer.model, trainer.is_distributed, half=True
    )
    print(summary)

    ap1, ap2, summary = trainer.evaluator.evaluate(
        trainer.model, trainer.is_distributed, half=True
    )
    print("Total:")
    print(summary)

    print(per_class_eval)
    print(ap1, ap2)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    main(exp, args)
