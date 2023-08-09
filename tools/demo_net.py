#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import time
import torch
import tqdm

from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer, _create_text_labels
from slowfast.utils.misc import get_class_names

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        start = time.time()
        if cfg.DEMO.THREAD_ENABLE:
            frame_provider = ThreadVideoManager(cfg)
        else:
            frame_provider = VideoManager(cfg)


        features = {"ids":[], "actions":[], }
        class_names, _, _ = get_class_names(cfg.DEMO.LABEL_FILE_PATH, None, None)
        common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )
        video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres= cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres= cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

        for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
            frame_provider.display(task)
            ##Todo: copy the code from last version
            ## but return the feature/label


            preds = task.action_preds
            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            n_instances = preds.shape[0]
            top_scores, top_classes = [], []
            for pred in preds:
                mask = pred >= video_vis.thres
                top_scores.append(pred[mask].tolist())
                top_class = torch.squeeze(torch.nonzero(mask), dim=-1).tolist()
                top_classes.append(top_class)
            # Create labels top k predicted classes with their scores.
            text_labels = []
            for i in range(n_instances):
                text_labels.append(
                    _create_text_labels(
                        top_classes[i],
                        top_scores[i],
                        class_names,
                        ground_truth = False,
                    )
                )
            features["ids"].append(task.id)
            if text_labels[0]:
                for label in text_labels[0][:]:
                    label_name = label.split('] ')[1]
                    features["actions"].append(label_name)
            else:
                features["actions"].append("None")
        # action_labels = features["actions"]
        # for i in range(len(action_labels)):
        #     action_labels[i] = action_labels[i].strip('"')
        # data = " ".join(action_labels)

        frame_provider.join()
        frame_provider.clean()
        logger.info("Finish demo in: {}".format(time.time() - start))
        return features
