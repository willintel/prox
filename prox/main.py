# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import open3d as o3d
import torch

import smplx


from misc_utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False

def main(**args):
    data_folder = args.get('recording_dir')
    recording_name = osp.basename(args.get('recording_dir'))
    scene_name = recording_name.split("_")[0]
    base_dir = os.path.abspath(osp.join(args.get('recording_dir')))
    if not osp.exists(osp.join(base_dir, "calibration")):
        base_dir = os.path.abspath(osp.join(args.get('recording_dir'), os.pardir, os.pardir))
    print("base_dir:", base_dir)
    keyp_dir = osp.join(base_dir, 'keypoints')
    keyp_folder = osp.join(keyp_dir, recording_name)
    if not osp.exists(keyp_folder):
        keyp_folder = keyp_dir
    cam2world_dir = osp.join(base_dir, 'cam2world')
    scene_dir = osp.join(base_dir, 'scenes')
    calib_dir = osp.join(base_dir, 'calibration')
    sdf_dir = osp.join(base_dir, 'sdf')
    body_segments_dir = osp.join(base_dir, 'body_segments')
    model_folder = args.get('model_folder')
    body_segments_dir = osp.join(model_folder, "body_segments")


    output_folder = args.get('output_folder')
    output_folder = osp.expandvars(output_folder)
    output_folder = osp.join(output_folder, recording_name)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)
    #remove 'output_folder' from args list
    args.pop('output_folder')

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    body_scene_rendering_dir = os.path.join(output_folder, 'renderings')
    if not osp.exists(body_scene_rendering_dir):
        os.mkdir(body_scene_rendering_dir)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    img_folder = args.pop('img_folder', 'Color')
    dataset_obj = create_dataset(img_folder=img_folder,data_folder=data_folder, keyp_folder=keyp_folder, calib_dir=calib_dir,**args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    # Create the camera object
    camera_center = None \
        if args.get('camera_center_x') is None or args.get('camera_center_y') is None \
        else torch.tensor([args.get('camera_center_x'), args.get('camera_center_y')], dtype=dtype).view(-1, 2)
    print("args.get('focal_length_x'):", args.get('focal_length_x'))
    camera = create_camera(focal_length_x=args.get('focal_length_x'),
                           focal_length_y=args.get('focal_length_y'),
                           center= camera_center,
                           batch_size=args.get('batch_size'),
                           dtype=dtype)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        camera = camera.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    for idx, data in enumerate(dataset_obj):

        img = data['img']
        fn = data['fn']
        keypoints = data['keypoints']
        depth_im = data['depth_im']
        mask = data['mask']
        init_trans = None if data['init_trans'] is None else torch.tensor(data['init_trans'], dtype=dtype).view(-1,3)
        scan = data['scan_dict']
        print('Processing: {}'.format(data['img_path']))

        curr_result_folder = osp.join(result_folder, fn)
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        curr_mesh_folder = osp.join(mesh_folder, fn)
        if not osp.exists(curr_mesh_folder):
            os.makedirs(curr_mesh_folder)
        #TODO: SMPLifyD and PROX won't work for multiple persons
        for person_id in range(keypoints.shape[0]):
            if person_id >= max_persons and max_persons > 0:
                continue

            curr_result_fn = osp.join(curr_result_folder,
                                      '{:03d}.pkl'.format(person_id))
            curr_mesh_fn = osp.join(curr_mesh_folder,
                                    '{:03d}.ply'.format(person_id))
            curr_body_scene_rendering_fn = osp.join(body_scene_rendering_dir, fn + '.png')

            curr_img_folder = osp.join(output_folder, 'images', fn,
                                       '{:03d}'.format(person_id))
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)

            if gender_lbl_type != 'none':
                if gender_lbl_type == 'pd' and 'gender_pd' in data:
                    gender = data['gender_pd'][person_id]
                if gender_lbl_type == 'gt' and 'gender_gt' in data:
                    gender = data['gender_gt'][person_id]
            else:
                gender = input_gender

            if gender == 'neutral':
                body_model = neutral_model
            elif gender == 'female':
                body_model = female_model
            elif gender == 'male':
                body_model = male_model

            out_img_fn = osp.join(curr_img_folder, 'output.png')

            fit_single_frame(img, keypoints[[person_id]], init_trans, scan,
                             cam2world_dir=cam2world_dir,
                             scene_dir=scene_dir,
                             sdf_dir=sdf_dir,
                             body_segments_dir=body_segments_dir,
                             scene_name=scene_name,
                             body_model=body_model,
                             camera=camera,
                             joint_weights=joint_weights,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             out_img_fn=out_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             body_scene_rendering_fn=curr_body_scene_rendering_fn,
                             shape_prior=shape_prior,
                             expr_prior=expr_prior,
                             body_pose_prior=body_pose_prior,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             angle_prior=angle_prior,
                             **args)
        break
        
    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_FOLDER="/media/psf/WVerbatim/data/mevolve/prox/quantitative/recordings/vicon_03301_01-1"
    MODELS_FOLDER="/media/psf/WVerbatim/data/mevolve/prox"
    argv=[
         "--config", SCRIPT_PATH+"/../cfg_files/SMPLifyD.yaml",
#        "--config", SCRIPT_PATH+"/../cfg_files/RGB.yaml",
        "--recording_dir", DATA_FOLDER,
        "--output_folder", DATA_FOLDER +"/output",
        "--vposer_ckpt", MODELS_FOLDER + "/models/vposer_v1_0/",
        "--part_segm_fn", MODELS_FOLDER + "/models/smplx_parts_segm.pkl",
        "--model_folder", MODELS_FOLDER + "/models",
        "--use_cuda", "0",
        "--interpenetration", "0",
        "--save_meshes", "1",
        "--render_results", "0",
        # "--camera_center_x", 
        ]
    
    MODELS_FOLDER="/media/psf/WVerbatim/data/mevolve/prox"
    PROX_SRC_PATH = "/home/william/dev/thirdparty/prox"
#    FIT_DATA_FOLDER = fdata.get_root_folder()
#    print("FIT_DATA_FOLDER:", FIT_DATA_FOLDER)
    FIT_DATA_FOLDER = "/media/psf/WVerbatim/data/mevolve/inhome-test-rig/2020_05_21/d435-dynamic-human_4shot-21_08_33/snapshots/fit-data"
#     argv = [
# #        "python3", os.path.join(PROX_SRC_PATH, "prox", "main.py"),
#        "--config", os.path.join(PROX_SRC_PATH,"cfg_files", "SMPLifyD-d435.yaml"),
#        # "--config", os.path.join(PROX_SRC_PATH,"cfg_files", "RGB-d435.yaml"),
# #        "--config", SCRIPT_PATH+"/../cfg_files/RGB.yaml",
#        "--recording_dir", FIT_DATA_FOLDER,
#        "--output_folder", os.path.join(FIT_DATA_FOLDER, "prox"),
#        "--vposer_ckpt", MODELS_FOLDER + "/models/vposer_v1_0/",
#        "--part_segm_fn", MODELS_FOLDER + "/models/smplx_parts_segm.pkl",
#        "--model_folder", MODELS_FOLDER + "/models",
#        "--use_cuda", "0",
#        "--interpenetration", "0",
#        "--save_meshes", "1",
#        "--render_results", "0",
#        # "--camera_center_x", 
#        ]
    
#    argv=None
    args = parse_config(argv)
#    print("args:", args)
    main(**args)
