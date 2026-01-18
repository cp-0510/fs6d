import os
import cv2
import sys
import json
import time
import torch
import warnings
import numpy as np
from PIL import Image
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from os.path import join as pjoin
from bop_toolkit_lib import inout

warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from lib import rendering, network
from dataset import LineMOD_Dataset
from evaluation import utils
from evaluation import config as cfg

gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["EGL_DEVICE_ID"] = str(gpu_id)
DEVICE = torch.device("cuda")

datapath = Path(cfg.DATA_PATH)
eval_dataset = LineMOD_Dataset.Dataset(datapath / "lm")


rcnnIdx_to_lmoIds_dict = {0: 1, 1: 5, 2: 6, 3: 8, 4: 9, 5: 10, 6: 11, 7: 12}
rcnn_cfg = get_cfg()
rcnn_cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
)
rcnn_cfg.MODEL.WEIGHTS = pjoin(
    base_path, "checkpoints", "lmo_fasterrcnn_model.pth"
)
rcnn_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
rcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
predictor = DefaultPredictor(rcnn_cfg)

cfg.DATASET_NAME = "lm"
cfg.RENDER_WIDTH = eval_dataset.cam_width
cfg.RENDER_HEIGHT = eval_dataset.cam_height
cfg.HEMI_ONLY = True

ckpt_file = pjoin(base_path, "checkpoints", "model.pth")
model_net = network.net().to(DEVICE)
model_net.load_state_dict(torch.load(ckpt_file), strict=True)
model_net.eval()

codebook_dir = pjoin(
    base_path,
    "evaluation/object_codebooks",
    cfg.DATASET_NAME,
    f"zoom_{cfg.ZOOM_DIST_FACTOR}",
    f"views_{cfg.RENDER_NUM_VIEWS}"
)

object_codebooks = utils.codebook_generation(
    codebook_dir=codebook_dir,
    model_func=model_net,
    dataset=eval_dataset,
    config=cfg,
    device=DEVICE
)
obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)

test_dir = datapath / "lmo" / "test"
eval_dir = pjoin(base_path, "evaluation/pred_results/LMO")
os.makedirs(eval_dir, exist_ok=True)

raw_results = []
runtime_list = []

for scene_id in sorted(os.listdir(test_dir)):
    scene_dir = pjoin(test_dir, scene_id)
    if not os.path.isdir(scene_dir):
        continue

    cam_info = json.load(open(pjoin(scene_dir, "scene_camera.json")))
    rgb_dir = pjoin(scene_dir, "rgb")
    depth_dir = pjoin(scene_dir, "depth")

    for rgb_name in sorted(os.listdir(rgb_dir)):
        if not rgb_name.endswith(".png"):
            continue

        view_id = int(rgb_name.split(".")[0])
        start_time = time.time()

        rgb_np = cv2.imread(pjoin(rgb_dir, rgb_name))
        rgb = torch.tensor(rgb_np, dtype=torch.float32)\
                    .permute(2, 0, 1)[None].to(DEVICE)

        depth = torch.tensor(
            np.array(Image.open(pjoin(depth_dir, rgb_name))),
            dtype=torch.float32
        ).to(DEVICE)
        depth *= cam_info[str(view_id)]["depth_scale"]
        depth *= cfg.MODEL_SCALING

        cam_K = torch.tensor(
            cam_info[str(view_id)]["cam_K"],
            dtype=torch.float32
        ).view(1, 3, 3).to(DEVICE)

        output = predictor(rgb_np)
        inst = output["instances"]
        cls_ids = inst.pred_classes
        boxes = inst.pred_boxes.tensor
        scores = inst.scores

        if len(cls_ids) == 0:
            continue

        for cls_id in torch.unique(cls_ids):
            lmo_obj_id = rcnnIdx_to_lmoIds_dict[cls_id.item()]
            obj_codebook = object_codebooks[lmo_obj_id]

            keep = (cls_ids == cls_id)
            tar_boxes = boxes[keep]
            tar_scores = scores[keep]

            if tar_boxes.shape[0] == 0:
                continue

            tar_rgbs, tar_depths, scales = utils.crop_resize(
                rgb, depth, tar_boxes
            )

            camKs = []
            for i, box in enumerate(tar_boxes):
                Kc = utils.camK_crop(cam_K, box)
                sx, sy = scales[i]
                camKs.append(utils.camK_resize(Kc, sx, sy))
            camKs = torch.cat(camKs, dim=0)

            pose_ret = utils.full_pose(
                model_func=model_net,
                obj_rgbs=tar_rgbs,
                obj_rcnn_scores=tar_scores,
                obj_codebook=obj_codebook,
                cam_K=camKs,
                config=cfg,
                device=DEVICE,
                obj_renderer=obj_renderer
            )

            raw_results.append({
                "scene_id": int(scene_id),
                "im_id": int(view_id),
                "obj_id": int(lmo_obj_id),
                "score": pose_ret["raw_score"].item(),
                "R": cfg.POSE_TO_BOP(
                    pose_ret["raw_R"]
                ).squeeze().cpu().numpy(),
                "t": (pose_ret["raw_t"] * 1000.0)
                        .squeeze().cpu().numpy()
            })

            runtime_list.append(pose_ret["raw_time"])

        if view_id % 100 == 0:
            print(
                f"scene {scene_id}, view {view_id}, "
                f"mean time {np.mean(runtime_list):.4f}s"
            )

    print(f"scene {scene_id} finished")

out_file = pjoin(
    eval_dir,
    f"view{cfg.RENDER_NUM_VIEWS}-vp{cfg.VP_NUM_TOPK}-lmo-fast-rcnn.csv"
)
inout.save_bop_results(out_file, raw_results)

print(
    f"Saved to {out_file}, "
    f"mean runtime: {np.mean(runtime_list):.4f}s"
)

del obj_renderer
