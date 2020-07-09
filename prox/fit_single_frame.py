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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img
import json
from optimizers import optim_factory

import fitting
from human_body_prior.tools.model_loader import load_vposer
from psbody.mesh import Mesh
import scipy.sparse as sparse


def fit_single_frame(img,
                     keypoints,
                     init_trans,
                     scan,
                     body_model,
                     body_segments_dir,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     shape_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     body_scene_rendering_fn='body_scene.png',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     data_weights=None,
                     body_pose_prior_weights=None,
                     shape_weights=None,
                     depth_loss_weight=1e2,
                     focal_length_x=5000.,
                     focal_length_y=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     ####################
                     ### PROX
                     render_results=True,
                     camera_mode='moving',
                     ## Depth
                     s2m=False,
                     s2m_weights=None,
                     m2s=False,
                     m2s_weights=None,
                     rho_s2m=1,
                     rho_m2s=1,
                     init_mode=None,
                     trans_opt_stages=None,
                     viz_mode='mv',
                     previous_result=None,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    body_model.reset_params()
    body_model.transl.requires_grad = True

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if visualize:
        pil_img.fromarray((img * 255).astype(np.uint8)).show()

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    if previous_result is not None:
        pose_embedding = previous_result['pose_embedding']
        body_mean_pose = previous_result['body_pose']

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    scan_tensor = None
    scan_normal = None
    keypoints3d = None
    if scan is not None:
        scan_tensor = torch.tensor(scan.get('points'), device=device, dtype=dtype).unsqueeze(0)
        if scan.get('normals') is not None:
            scan_normal = torch.tensor(scan.get('normals'), device=device, dtype=dtype).unsqueeze(0)
        keypoints3d = torch.tensor(scan.get('keypoints3d'), device=device, dtype=dtype).unsqueeze(0)
    
    cam2world = np.eye(4)
    R = torch.tensor(cam2world[:3, :3].reshape(3, 3), dtype=dtype, device=device)
    t = torch.tensor(cam2world[:3, 3].reshape(1, 3), dtype=dtype, device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    
    if s2m:
        opt_weights_dict['s2m_weight'] = s2m_weights
    if m2s:
        opt_weights_dict['m2s_weight'] = m2s_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # load indices of the head of smpl-x model
    with open( osp.join(body_segments_dir, 'body_mask.json'), 'r') as fp:
        head_indx = np.array(json.load(fp))
    N = body_model.get_num_verts()
    body_indx = np.setdiff1d(np.arange(N), head_indx)
    head_mask = np.in1d(np.arange(N), head_indx)
    body_mask = np.in1d(np.arange(N), body_indx)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')

    init_t = init_trans

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      camera_mode=camera_mode,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               s2m=s2m,
                               m2s=m2s,
                               rho_s2m=rho_s2m,
                               rho_m2s=rho_m2s,
                               head_mask=head_mask,
                               body_mask=body_mask,
                               R=R,
                               t=t,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, viz_mode=viz_mode, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape
        print("camera_mode:", camera_mode, " init_t:", init_t)
        body_model.reset_params(body_pose=body_mean_pose, transl=init_t)
        camera_opt_params = [body_model.transl, body_model.global_orient]

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)

        def export_body_model(body_model, fn):
            # return
            try:
                import trimesh
                body_pose = vposer.decode(
                    pose_embedding,
                    output_type='aa').view(1, -1) if use_vposer else None
                append_wrists = kwargs.get('model_type', 'smpl') == 'smpl' and use_vposer
                if append_wrists:
                        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                                 dtype=body_pose.dtype,
                                                 device=body_pose.device)
                        body_pose = torch.cat([body_pose, wrist_pose], dim=1)
        
                model_output = body_model(return_verts=True, body_pose=body_pose)
                vertices_np = model_output.vertices.detach().cpu().numpy().squeeze()
                body = trimesh.Trimesh(vertices_np, body_model.faces, process=False)
                body.export(fn)
            finally: 
                pass

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            scan_tensor=scan_tensor,
            scan_normal=scan_normal,
            keypoints3d=keypoints3d,
            return_full_pose=False, return_verts=False)

        export_body_model(body_model, "./body_model-cam_before.ply")

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        # body_model.reset_params(transl=init_t)
        
        export_body_model(body_model, "./body_model-cam_after.ply")

        if interactive:
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []
        body_transl = body_model.transl.clone().detach()
        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(transl=body_transl,
                                     global_orient=orient,
                                     body_pose=body_mean_pose)
            body_model.reset_params(**new_params)
            if use_vposer and previous_result is None:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
                if opt_idx not in trans_opt_stages:
                    body_model.transl.requires_grad = False
                else:
                    body_model.transl.requires_grad = True
                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                loss.reset_loss_weights(curr_weights)

                # only take 10000 points in scan
                import random
                _, n, _ = scan_tensor.shape
                ids = list(range(n))
                random.shuffle(ids)
                max_n = min(n, n)
                ids = ids[:max_n]
                
                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, 
                    gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    scan_tensor=scan_tensor[:, ids, :],
                    scan_normal=  None if (scan_normal is None) else scan_normal[:, ids, :],
                    keypoints3d=keypoints3d,
                    return_verts=True, return_full_pose=True,
                    opt_idx=opt_idx)

                if interactive:
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding,
                    vposer=vposer,
                    use_vposer=use_vposer)

                export_body_model(body_model, "./body_model-or_idx{}-opt_idx{}.ply".format(or_idx, opt_idx))
                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['pose_embedding'] = pose_embedding.detach().cpu().numpy()
                body_pose = vposer.decode(
                    pose_embedding,
                    output_type='aa').view(1, -1) if use_vposer else None
                result['body_pose'] = body_pose.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})


        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

        print("save_meshes:", save_meshes)
        save_meshes=True
        if save_meshes or visualize:
            body_pose = vposer.decode(
                pose_embedding,
                output_type='aa').view(1, -1) if use_vposer else None
    
            model_type = kwargs.get('model_type', 'smpl')
            append_wrists = model_type == 'smpl' and use_vposer
            if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
    
            model_output = body_model(return_verts=True, body_pose=body_pose)
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()
    
            import trimesh
            print("vertices:", vertices.shape, " mesh_fn:", mesh_fn)
            out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
            out_mesh.export(mesh_fn)

    return {'body_pose': body_pose,

            'camera' : camera,

            'body_model' : pickle.loads(pickle.dumps(body_model)),
            'pose_embedding': pose_embedding.detach().cpu().numpy().copy(),

            'gt_joints': gt_joints.detach().cpu().numpy().copy(),
            'joint_weights': joint_weights,

            'scan_tensor': scan_tensor.detach().cpu().numpy().copy(),
            'scan_normal': scan_normal.detach().cpu().numpy().copy(),
            's2m_weights': s2m_weights
            }