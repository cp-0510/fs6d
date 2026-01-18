# Few-Shot 6D Object Pose Estimation via Decoupled Rotation and Translation with Viewpoint Encoding
<p align="center">
    <img src ="assets/overview.png" width="800" />
</p>

## Requirements
* Please start by installing [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) with Pyhton3.8 or above.
* This project requires the evaluation code from [bop_toolkit](https://github.com/thodan/bop_toolkit) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit).
* Install `detectron2` from [source](https://github.com/facebookresearch/detectron2)

## Dataset
Our evaluation is conducted on three datasets all downloaded from [BOP website](https://bop.felk.cvut.cz/datasets). All three datasets are stored in the same directory. e.g. ``Dataspace/lm, Dataspace/lmo, Dataspace/ycb``.

## Rotation Estimation
* Training 
The rotation estimation network is trained using ShapeNet. Before training, please preprocess ShapeNet using the provided script:``training/preprocess_shapenet.py``.(Note: This preprocessing step requires Blender for rendering. Please ensure Blender is correctly installed.)
* Evaluating 
The trained rotation estimation network is evaluated on the LINEMOD and Occluded LINEMOD datasets using the provided evaluation pipelines: ``python LM_pipeline.py`` ``python LMO_pipeline.py`` 

## Translation Estimation
The overall design of the translation estimation module is largely inspired by GDRNet.
* Training 
``./core/modeling/train_model.sh <config_path> <gpu_ids> (other args)``
* Evaluating 
``./core/modeling/test_model.sh <config_path> <gpu_ids> <ckpt_path> (other args)``

