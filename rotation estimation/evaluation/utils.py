import os
import time
import glob
import math
import torch
import numpy as np
import torch.nn.functional as F
from lib import rendering
import torch
import torchvision.transforms as transforms
from PIL import Image

def rotation_to_position(R):
    t = torch.tensor([0, 0, 1], dtype=torch.float32, device=R.device)[None, ..., None]
    pos = (-R.squeeze().transpose(-2, -1) @ t).squeeze()
    return pos

def rotation_error(R0, R1):
    cos = (torch.trace(R0.squeeze().clone() @ R1.squeeze().T) - 1.0) / 2.0
    if cos < -1:
        cos = -2 - cos
    elif cos > 1:
        cos = 2 - cos
    return torch.arccos(cos)/math.pi*180

def crop_resize(rgb, depth, boxes):
    rgbs, depths, scales = [], [], []

    for box in boxes:
        x1, y1, x2, y2 = box.int()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[-1], x2), min(rgb.shape[-2], y2)

        crop_rgb = rgb[:, :, y1:y2, x1:x2]
        crop_dep = depth[y1:y2, x1:x2]

        h, w = crop_dep.shape
        sx = 128.0 / w
        sy = 128.0 / h
        scales.append([sx, sy])

        crop_rgb = F.interpolate(
            crop_rgb, size=(128, 128),
            mode='bilinear', align_corners=False
        )
        crop_dep = F.interpolate(
            crop_dep[None, None],
            size=(128, 128),
            mode='nearest'
        ).squeeze(0).squeeze(0)

        rgbs.append(crop_rgb)
        depths.append(crop_dep)

    return torch.cat(rgbs, 0), torch.stack(depths, 0), torch.tensor(scales, device=DEVICE)

def camK_crop(K, box):
    Kc = K.clone()
    Kc[0, 2] -= box[0]
    Kc[1, 2] -= box[1]
    return Kc

def camK_resize(K, sx, sy):
    Kr = K.clone()
    Kr[0, 0] *= sx
    Kr[1, 1] *= sy
    Kr[0, 2] *= sx
    Kr[1, 2] *= sy
    return Kr

def viewpoint_sampling_and_encoding(model_func, obj_model_file, obj_diameter, config, intrinsic, device):
    render_width = config.RENDER_WIDTH
    render_height = config.RENDER_HEIGHT
    render_num_views = config.RENDER_NUM_VIEWS
    render_obj_scale = config.MODEL_SCALING
    store_featmap = config.SAVE_FTMAP
     
    obj_Rs = rendering.evenly_distributed_rotation(n=render_num_views, random_seed=config.RANDOM_SEED)

    if config.HEMI_ONLY:
        upper_hemi = obj_Rs[:, 2, 2] > 0
        hemi_obj_Rs = obj_Rs[upper_hemi]
        obj_Rs = hemi_obj_Rs
    
    obj_codebook_Z_vec = list()
    obj_codebook_Z_map = list()
    Rs_chunks = torch.split(obj_Rs, split_size_or_sections=config.VIEWBOOK_BATCHSIZE, dim=0)

    render_costs = list()
    infer_costs = list()
    intrinsic = intrinsic.to(device)

    obj_mesh, _ = rendering.load_object(obj_model_file, resize=False, recenter=False)
    obj_mesh.rescale(scale=render_obj_scale)

    for cbk_Rs in Rs_chunks:
        render_timer = time.time()
        cbk_depths, cbk_masks, cbk_rgbs = rendering.rendering_views(obj_mesh=obj_mesh,
                                                        intrinsic=intrinsic.cpu(),
                                                        R=cbk_Rs,
                                                        width=render_width,
                                                        height=render_height)
        render_cost = time.time() - render_timer
        render_costs.append(render_cost)
        
        encoder_timer = time.time()

        cbk_depths, cbk_masks, cbk_rgbs = cbk_depths.to(device), cbk_masks.to(device), cbk_rgbs.to(device)

        extrinsic = torch.ones((cbk_Rs.size(0), 4, 4), device=cbk_Rs.device)
        extrinsic[:, :3, :3] = cbk_Rs

        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])
        to_pil = transforms.ToPILImage()
        cbk_zoom_rgbs = torch.stack([
            transform(to_pil(cbk_rgbs[i].permute(2, 0, 1).byte()))
            for i in range(cbk_rgbs.size(0))
        ])
        cbk_zoom_rgbs = cbk_zoom_rgbs.to(device)

        with torch.no_grad():
            z_map, z_vec = model_func.vipri_encoder(cbk_zoom_rgbs, return_maps=True)
        
        obj_codebook_Z_vec.append(z_vec.detach().cpu())
        if store_featmap:
            obj_codebook_Z_map.append(z_map.detach().cpu())

    del cbk_depths, cbk_masks, z_map, z_vec
    encoder_cost = time.time() - encoder_timer
    infer_costs.append(encoder_cost)

    obj_codebook_Z_vec = torch.cat(obj_codebook_Z_vec, dim=0)
    if store_featmap:
        obj_codebook_Z_map = torch.cat(obj_codebook_Z_map, dim=0)
    else:
        obj_codebook_Z_map = None
    print('render_time:{:.3f}, encoding_time:{:.3f}'.format(np.sum(render_costs), np.sum(infer_costs)))
    return {"Rs":obj_Rs,
            "diameter":obj_diameter,
            "Z_vec":obj_codebook_Z_vec,
            "Z_map":obj_codebook_Z_map,
            "obj_mesh":obj_mesh,
           }

def codebook_generation(model_func, codebook_dir, dataset, config, device):
    object_codebooks = dict()

    codebook_files = sorted(glob.glob(os.path.join(codebook_dir, 
                            '*_views_{}.npy'.format(config.RENDER_NUM_VIEWS))))
    intrinsic = dataset.cam_K
    obj_model_files = dataset.obj_model_file
    obj_diameter_info = dataset.obj_diameter

    num_objects = len(obj_model_files)

    if len(codebook_files) == num_objects:
        for obj_cbk_file in codebook_files:
            cbk_name = obj_cbk_file.split('/')[-1]
            obj_id = int(cbk_name.split('_')[-3])
            print('Loading ', obj_cbk_file)
            with open(obj_cbk_file, 'rb') as f:
                object_codebooks[obj_id] = np.load(f, allow_pickle=True).item()
    else:
        print('generating codebook for {} viewpoints ...'.format(config.RENDER_NUM_VIEWS))
        for obj_id, obj_model_file in obj_model_files.items():
            obj_diameter = obj_diameter_info[obj_id] * config.MODEL_SCALING
            obj_cbk = viewpoint_sampling_and_encoding(model_func=model_func,
                                                            obj_model_file=obj_model_file,
                                                            obj_diameter=obj_diameter,
                                                            config=config,
                                                            intrinsic=intrinsic,
                                                            device=device)                                              
            if not os.path.exists(codebook_dir):
                os.makedirs(codebook_dir)
            codebook_file = os.path.join(codebook_dir, 
                            '{}_obj_{:02d}_views_{}.npy'.format(
                            config.DATASET_NAME, obj_id, config.RENDER_NUM_VIEWS))
                           
            with open(codebook_file, 'wb') as f:
                np.save(f, obj_cbk)
            object_codebooks[obj_id] = obj_cbk
            
            print('obj_id: ', obj_id, time.strftime('%m_%d-%H:%M:%S', time.localtime()))
    return object_codebooks

def full_pose(model_func, obj_rgbs, obj_rcnn_scores, obj_codebook, cam_K, config, obj_renderer, device, return_rcnn_idx=False):
    pose_ret = dict()
    obj_rgbs = obj_rgbs.to(device)
    zoom_test_rgbs = (
        0.299 * obj_rgbs[:, 0:1, :, :] +
        0.587 * obj_rgbs[:, 1:2, :, :] +
        0.114 * obj_rgbs[:, 2:3, :, :]
    )
    rot_timer = time.time()
    estimated_R, estimated_scores, estimated_rcnn_idx = \
        rotation_estimation(
            input_rgb=zoom_test_rgbs,
            rcnn_score=obj_rcnn_scores,
            model_net=model_func,
            object_codebook=obj_codebook,
            cfg=config
        )
    rot_cost = time.time() - rot_timer
    obj_raw_R = estimated_R[0]
    obj_score = estimated_scores[0]

    pose_ret['rcnn_idx'] = estimated_rcnn_idx
    pose_ret['raw_R'] = obj_raw_R
    pose_ret['raw_score'] = obj_score
    pose_ret['rot_time'] = rot_cost

    if return_rcnn_idx:
        return pose_ret, estimated_rcnn_idx

    return pose_ret

def rotation_estimation(input_rgb, rcnn_score, model_net, object_codebook, cfg):
    obj_codebook_Rs = object_codebook['Rs']
    obj_codebook_Z_vec = object_codebook['Z_vec']
    obj_codebook_Z_map = object_codebook['Z_map']
    
    with torch.no_grad():
        obj_query_z_map, obj_query_z_vec = model_net.vipri_encoder(input_rgb, return_maps=True)

    vp_obj_cosim_scores = F.cosine_similarity(obj_codebook_Z_vec.unsqueeze(0).to(obj_query_z_vec.device),
                                                obj_query_z_vec.unsqueeze(1), dim=2)
    mean_vp_obj_cosim_scores = vp_obj_cosim_scores.topk(k=1, dim=1)[0].mean(dim=1).squeeze()
    fused_obj_scores = mean_vp_obj_cosim_scores * rcnn_score.to(mean_vp_obj_cosim_scores.device)

    best_rcnn_idx = fused_obj_scores.max(dim=0)[1]
    best_obj_cosim_scores = vp_obj_cosim_scores[best_rcnn_idx]
    best_obj_query_z_map = obj_query_z_map[best_rcnn_idx][None, ...]
    topK_cosim_idxes = best_obj_cosim_scores.topk(k=cfg.VP_NUM_TOPK)[1]

    estimated_scores = best_obj_cosim_scores[topK_cosim_idxes]
    retrieved_codebook_R = obj_codebook_Rs[topK_cosim_idxes].clone()
    top_codebook_z_maps = obj_codebook_Z_map[topK_cosim_idxes, ...]
    
    with torch.no_grad():
        query_theta, pd_conf = model_net.inference(top_codebook_z_maps.to(best_obj_query_z_map.device), 
                                                    best_obj_query_z_map.expand_as(top_codebook_z_maps))
    stn_theta = F.pad(query_theta[:, :2, :2].clone(), (0, 1))
    
    homo_z_R = F.pad(stn_theta, (0, 0, 0, 1))
    homo_z_R[:, -1, -1] = 1.0
    estimated_xyz_R = homo_z_R @ retrieved_codebook_R.to(homo_z_R.device)
    pd_conf = pd_conf.squeeze(1)
    sorted_idxes = pd_conf.topk(k=len(pd_conf))[1]

    final_R = estimated_xyz_R[sorted_idxes][:cfg.POSE_NUM_TOPK]
    final_S = estimated_scores[sorted_idxes][:cfg.POSE_NUM_TOPK]
   
    return final_R.cpu(), final_S.cpu(), best_rcnn_idx
 

