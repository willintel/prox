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

import os.path as osp
import cv2
import numpy as np
import json


def compute_normals(depth_im, points, valid_idx, min_x, max_x, min_y, max_y, offset=3):
    h = depth_im.shape[0]
    w = depth_im.shape[1]
    def get_vec_2f1(uv1, uv2):
        idx1 = w*uv1[1] + uv1[0]
        idx2 = w*uv2[1] + uv2[0]
        p1 = points[idx1]
        p2 = points[idx2]
        if p1[2] < 0.1 or p2[2] < 0.1:
            return None
        return p2 - p1
    print("w:", w, " h:", h)
    normals = np.zeros(points.shape, dtype=float)
    for y in range(min_y+offset+1, max_y-offset-1):
        for x in range(min_x+offset+1, max_x-offset-1):
            idx = w*y + x
            if valid_idx[idx] == False:
                continue
            lr = get_vec_2f1([x-offset, y],  [x+offset, y])
            tb = get_vec_2f1([x, y-offset], [x, y+offset])
            if lr is None or tb is None:
                continue 
            lr = lr/(np.linalg.norm(lr)+1.e-9)
            tb = tb/(np.linalg.norm(tb)+1.e-9)
            normal = np.cross(tb, lr)
            if normal[2] > -1.e-3:
                continue
            normals[idx] = normal
    return normals

class Projection():
    def __init__(self, calib_dir):
        depth_fn = osp.join(calib_dir, 'depth.json')
        if not osp.exists(depth_fn):
            depth_fn = osp.join(calib_dir, 'IR.json')

        color_fn = osp.join(calib_dir, 'color.json')
        if not osp.exists(color_fn):
            color_fn = osp.join(calib_dir, 'Color.json')

        with open(depth_fn, 'r') as f:
            self.depth_cam = json.load(f)
        with open(color_fn, 'r') as f:
            self.color_cam = json.load(f)

    def row(self, A):
        return A.reshape((1, -1))
    def col(self, A):
        return A.reshape((-1, 1))

    def unproject_depth_image(self, depth_image, cam):
        us = np.arange(depth_image.size) % depth_image.shape[1]
        vs = np.arange(depth_image.size) // depth_image.shape[1]
        ds = depth_image.ravel()
        uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
        #unproject
        xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                      np.asarray(cam['camera_mtx']), np.asarray(cam['k']))
        xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), self.col(uvd[:, 2])))
        xyz_camera_space[:, :2] *= self.col(xyz_camera_space[:, 2])  # scale x,y by z
        other_answer = xyz_camera_space - self.row(np.asarray(cam['view_mtx'])[:, 3])  # translate
        xyz = other_answer.dot(np.asarray(cam['view_mtx'])[:, :3])  # rotate

        return xyz.reshape((depth_image.shape[0], depth_image.shape[1], -1))

    def get_valid_value(self, uv, image, radius_factors=range(1,8)):
        directions = [
                        [0, 0],
                        [0, 1],
                        [-1, 1],
                        [-1, 0],
                        [-1, -1],
                        [-1, 0],
                        [1, -1],
                        [1, 0],
                        [1, 1]
                     ]
        w = image.shape[1]
        h = image.shape[0]
        result = None
        for factor in radius_factors:
            for d in directions:
                x = int(uv[0]) + d[0]*factor
                y = int(uv[1]) + d[1]*factor
                if x <= 0 or x >= w or y <= 0 or y >= h:
                    continue
                value = image[y, x]
                if value <= 0.1:
                    continue
                result = value
                break
        return result
    
    def projectPoints(self, v, cam):
        v = v.reshape((-1,3)).copy()
        return cv2.projectPoints(v, np.asarray(cam['R']), np.asarray(cam['T']), np.asarray(cam['camera_mtx']), np.asarray(cam['k']))[0].squeeze()

    def create_scan(self, mask, depth_im, color_im=None, mask_on_color=False, coord='color', TH=1e-2, default_color=[1.00, 0.75, 0.80], keypoints=None):
        # creating mask from bounding box of keypoints 
        if color_im is not None and keypoints is not None :
            min_x = 1.e6
            min_y = 1.e6
            max_x = 0
            max_y = 0

            ankle_ids = [19, 20, 21, 22, 23, 24] # skip ankles by default
            kps = np.round(keypoints[0].copy()).astype(int)
            for i,kp in enumerate(kps):
                if i in ankle_ids:
                    continue
                if kp[2] <= 0:
                    continue
                min_x = min(kp[0], min_x)
                max_x = max(kp[0], max_x)
                min_y = min(kp[1], min_y)
                max_y = max(kp[1], max_y)
            margin = 0
            min_x = max(min_x-margin, 0)
            min_y = max(min_y-margin, 0)
            max_x = min(max_x+margin, color_im.shape[1])
            max_y = min(max_y+margin, color_im.shape[0])
        else:
            min_x = 0
            min_y = 0
            max_x = color_im.shape[1]
            max_y = color_im.shape[0]
        
        if False:
            roi_mask = np.ones(color_im.shape, dtype=np.uint8)*255
            roi_mask[min_y:max_y, min_x:max_x] = 0

            print("keypoints:", keypoints)
            print("min_x:", min_x, " max_x:", max_x, " min_y:", min_y, " max_y:", max_y)
            cv2.imshow("color_im", color_im)
            cv2.imshow("roi_mask", roi_mask)
            cv2.waitKey()
    

        if not mask_on_color and mask is not None:
            depth_im[mask != 0] = 0
        if depth_im.size == 0:
            return {'v': []}

        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        colors = np.tile(default_color, [points.shape[0], 1])

        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= min_y, uvs[:, 1] < max_y)
        valid_y = np.logical_and(uvs[:, 0] >= min_x, uvs[:, 0] < max_x)
        valid_idx = np.logical_and(valid_x, valid_y)
        
        normals = compute_normals(depth_im, points, valid_idx, min_x, max_x, min_y, max_y, offset=3)
        valid_idx = normals[:,2] < 0.0

        keypoints3d = []
        kp_uvs = []
        kp_ids = []
        if color_im is not None and keypoints is not None:
            oncolor_index_im = np.zeros(color_im.shape[:2], dtype=int)
            for i, v in enumerate(valid_idx):
                if v == False: 
                    continue
                uv = uvs[i]
                oncolor_index_im[uv[1], uv[0]] = i
            kps = np.round(keypoints[0].copy()).astype(int)

            # removing keypoints if side views
            right_ids = [15, 17, 2, 3, 4, 9, 10, 11, 22, 23, 24, 1, 8]
            left_ids = [16, 18, 5, 6, 7, 12, 13, 14, 19, 20, 21, 1, 8]
            skip_ids = [19, 20, 21, 22, 23, 24] # skip ankles by default
            if kps[15][2] > 0 and kps[16][2] > 0:
                print("detected front side")
            elif kps[15][2] < 1 and kps[16][2] < 1:
                print("detected back side")
            elif kps[15][2] < 1 and kps[17][2] < 1:
                print("detected left side")
                skip_ids = skip_ids + right_ids
            elif kps[16][2] < 1 and kps[18][2] < 1:
                print("detected right side")
                skip_ids = skip_ids + left_ids
            print("skip_ids:", skip_ids)

            keypoints3d = np.zeros(kps.shape)
            for i,uv in enumerate(kps):
                if i in skip_ids: # skip ankles
                    continue
                if kps[i][2] <= 0.1:
                    continue
                value = self.get_valid_value(uv, oncolor_index_im)
                if value is None:
                    continue
                if points[value, 2] <= 0.1:
                    continue
                keypoints3d[i] = points[value, :]
                kp_ids.append(i)
                # checking not in face area
                if i not in [0, 15, 16, 17, 18]:
                    keypoints3d[i] *= 1.05 # adding depth
                    #keypoints3d[i][2] += 0.05 # adding 5cm depth
                kp_uvs.append(uv)


        print("kp_ids:", kp_ids)
        print("keypoints3d:", keypoints3d)
        print("kps:", kps)
            
        if mask_on_color:
            valid_mask_idx = valid_idx.copy()
            valid_mask_idx[valid_mask_idx == True] = mask[uvs[valid_idx == True][:, 1], uvs[valid_idx == True][:,
                                                                                        0]] == 0
            uvs = uvs[valid_mask_idx == True]
            points = points[valid_mask_idx]
            normals = normals[valid_mask_idx]
            colors = np.tile(default_color, [points.shape[0], 1])
            # colors = colors[valid_mask_idx]
            valid_idx = valid_mask_idx
            if color_im is not None:
                colors[:, :3] = color_im[uvs[:, 1], uvs[:, 0]] / 255.0
        else:
            uvs = uvs[valid_idx == True]
            if color_im is not None:
                colors[valid_idx == True,:3] = color_im[uvs[:, 1], uvs[:, 0]]/255.0
            points = points[valid_idx]
            normals = normals[valid_idx]
            colors = colors[valid_idx]

        if coord == 'color':
            # Transform to color camera coord
            T = np.concatenate([np.asarray(self.color_cam['view_mtx']), np.array([0, 0, 0, 1]).reshape(1, -1)])
            stacked = np.column_stack((points, np.ones(len(points)) ))
            points = np.dot(T, stacked.T).T[:, :3]
            points = np.ascontiguousarray(points)

            # Transform/rotate normals to color camera coord
            Rot = np.eye(4)
            Rot[:3,:3] = T[:3,:3]
            stacked_normals = np.column_stack((normals, np.ones(len(normals)) ))
            normals = np.dot(Rot, stacked_normals.T).T[:, :3]
            normals = np.ascontiguousarray(normals)

        ind = points[:, 2] > TH

        return {'points':points[ind], 'normals' : normals[ind], 'colors':colors[ind], 'keypoints3d': keypoints3d}


    def align_color2depth(self, depth_im, color_im, interpolate=True):
        (w_d, h_d) = (512, 424)
        if interpolate:
            # fill depth holes to avoid black spots in aligned rgb image
            zero_mask = np.array(depth_im == 0.).ravel()
            depth_im_flat = depth_im.ravel()
            depth_im_flat[zero_mask] = np.interp(np.flatnonzero(zero_mask), np.flatnonzero(~zero_mask),
                                                 depth_im_flat[~zero_mask])
            depth_im = depth_im_flat.reshape(depth_im.shape)

        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]
        aligned_color = np.zeros((h_d, w_d, 3)).astype(color_im.dtype)
        aligned_color[valid_idx.reshape(h_d, w_d)] = color_im[uvs[:, 1], uvs[:, 0]]

        return aligned_color

    def align_depth2color(self, depth_im, depth_raw):
        (w_rgb, h_rgb) = (1920, 1080)
        (w_d, h_d) = (512, 424)
        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]

        aligned_depth = np.zeros((h_rgb, w_rgb)).astype('uint16')
        aligned_depth[uvs[:, 1], uvs[:, 0]] = depth_raw[valid_idx.reshape(h_d, w_d)]

        return aligned_depth
