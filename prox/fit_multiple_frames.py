import torch
import torch.optim as optim
# from torch.autograd import Variable
import torch.nn as nn
from torchviz import make_dot
import numpy as np

from psbody.mesh.visibility import visibility_compute
from psbody.mesh import Mesh

from human_body_prior.tools.model_loader import load_vposer
import misc_utils as utils

import icp

def calc_mean_betas(results):
    mean_betas = torch.zeros(results[0]['body_model'].betas.shape, dtype=torch.float, device=device)
    for e in results:
        mean_betas += e['body_model'].betas.clone()

    mean_betas /= len(results)
    return mean_betas


def parameterize_results(results):
    common_betas = calc_mean_betas(results)
    common_betas = torch.nn.Parameter(common_betas)

    for e in results:
        e['body_model'].betas = common_betas
        # e['body_model'].betas.requires_grad = False
        # e['body_model'].transl.requires_grad = True
        # e['body_model'].global_orient.requires_grad = False
        e['pose_embedding'] = torch.tensor(e['pose_embedding'], requires_grad=True, dtype=torch.float, device=device)

        e['gt_joints'] = torch.tensor(e['gt_joints'], requires_grad=False, dtype=torch.float, device=device)

        e['scan_tensor'] = torch.tensor(e['scan_tensor'], requires_grad=False, dtype=torch.float, device=device)
        e['scan_normal'] = torch.tensor(e['scan_normal'], requires_grad=False, dtype=torch.float, device=device)

    pass

def get_body_model_output(body_model, pose_embedding, vposer):
    body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)

    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
    body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    body_model_output = body_model(return_verts=True,
                                   body_pose=body_pose,
                                   return_full_pose=True)
    return body_model_output

def get_final_params(results):
    final_params = []
    for e in results:
        body_params = list(e['body_model'].parameters())
        body_params = list(filter(lambda x: x.requires_grad, body_params))
        final_params += body_params
        final_params.append(e['pose_embedding'])
    return final_params

@torch.no_grad()
def export_body_model(body_model, pose_embedding, vposer, fn):
    # return
    try:
        import trimesh
        bm_output = get_body_model_output(body_model, pose_embedding, vposer)
        
        vertices_np = bm_output.vertices.detach().cpu().numpy().squeeze()
        body = trimesh.Trimesh(vertices_np, body_model.faces, process=False)
        body.export(fn)
    finally: 
        pass

def export_pointcloud(scan_tensor, fn):
    import trimesh
    m = trimesh.Trimesh(scan_tensor[0].detach().cpu().numpy(), None, process=False)
    m.export(fn)
    pass

def calc_loss(results, vposer):
    total_joint_loss = 0.0
    total_s2m_loss = 0.0

    joint_robustifier = utils.GMoF(rho=100)
    s2m_robustifier = utils.GMoF(rho=5e-1)
    for e in results:
        camera = e['camera']
        body_model = e['body_model']
        joint_weights = e['joint_weights']
        weights = joint_weights.unsqueeze(dim=-1)
        gt_joints = e['gt_joints']
        pose_embedding = e['pose_embedding']

        scan_tensor = e['scan_tensor']
        scan_normal = e['scan_normal']
        s2m_weights = e['s2m_weights']

        body_model_output = get_body_model_output(body_model, pose_embedding, vposer)

        projected_joints = camera(body_model_output.joints)
        # Calculate the weights for each joints

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        # print("gt_joints:", gt_joints)
        # print("projected_joints:", projected_joints)
        # diff = gt_joints - projected_joints
        # print("diff:", torch.sum(diff**2))
        joint_diff = joint_robustifier(gt_joints - projected_joints)
        # print("joint_diff:", torch.sum(joint_diff))
        for i in range(gt_joints.shape[1]):
            if gt_joints[0,i,0].numpy() == 0.0 or gt_joints[0,i,1].numpy() == 0.0:
                joint_diff[0,i,:] = torch.tensor([0.0, 0.0])
        joint_loss = torch.sum(weights ** 2 * joint_diff)

        total_joint_loss += joint_loss

        vertices_np = body_model_output.vertices.detach().cpu().numpy().squeeze()
        body_faces_np = body_model.faces.reshape(-1, 3)
        m = Mesh(v=vertices_np, f=body_faces_np)

        (vis, n_dot) = visibility_compute(v=m.v, f=m.f, cams=np.array([[0.0, 0.0, 0.0]]))
        vis = vis.squeeze()

        s2m_dist = icp.dist_icp(scan_tensor, 
                               body_model_output.vertices[:, np.where(vis > 0)[0], :],
                               src_normal=scan_normal)
        s2m_dist = s2m_robustifier(s2m_dist) #(icp_dist.sqrt())
        s2m_dist = s2m_weights[-1] * torch.sum(s2m_dist)

        total_s2m_loss += s2m_dist

    print("total_joint_loss:", total_joint_loss, " total_s2m_loss:", total_s2m_loss)
    return (total_joint_loss + total_s2m_loss)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fn = "/home/william/dev/thirdparty/prox/separate_fit_result-4frames-2020_07_15.torch.bin"
    print("Load separate fit results from ({})".format(fn))
    results = torch.load(fn)
    parameterize_results(results)

    MODELS_FOLDER="/media/psf/Home/data/mevolve/prox"
    vposer_ckpt = MODELS_FOLDER + "/models/vposer_v1_0/"

    print("Load vposer from ({})".format(vposer_ckpt))
    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
    vposer = vposer.to(device=device)
    vposer.eval()

    lr = 1.0 #1e-3
    maxiters = 20
    n_epochs = 100

    params = get_final_params(results)
    # NOTE: SGB doesn't work!!! Loss keeps increasing
    # optimizer = optim.SGD(params, lr=lr, momentum=0.9,
    #                       weight_decay=0.0,
    #                       nesterov=True)
    # optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999),
    #                        weight_decay=0.0)

    maxiters = 20
    n_epochs = 10
    optimizer = optim.LBFGS(params, lr=lr, max_iter=maxiters)

    def closure():
        optimizer.zero_grad()
        loss = calc_loss(results, vposer)
        loss.backward()
        return loss

    for epoch in range(n_epochs):
        print("\n\nEpoch{} ============================================================".format(epoch))
        # loss = calc_loss(results, vposer)
        # loss.backward()    
        optimizer.step(closure)


        for n in range(len(results)):
            e = results[n]
            bm = e['body_model']
            print("transl:", bm.transl, " grad:", bm.transl.grad)
            print("global_orient:", bm.global_orient, " grad:", bm.global_orient.grad)
            fn = "fit_multiple_farmes-epoch{}_frame{}.ply".format(epoch, n)
            print("Export mesh to ({})".format(fn))
            export_body_model(e['body_model'], e['pose_embedding'], vposer, fn)
            fn = "fit_multiple_farmes-epoch{}_frame{}-pointcloud.ply".format(epoch, n)
            export_pointcloud(e['scan_tensor'], fn)
        
        # optimizer.zero_grad()
        # break
    

