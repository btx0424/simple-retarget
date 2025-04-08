import os
import torch
import torch.nn as nn
import hydra
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET

from smplx import SMPL
from scipy.spatial.transform import Rotation as sRot

CONDIF_PATH = os.path.join(os.path.dirname(__file__), "../cfg")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]


def build_chain(cfg) -> pk.Chain:
    mjcf_path = cfg.mjcf_path

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # remove the free joint of the base link
    root_name = cfg.get("root_name", "pelvis")
    root_body = root.find(f".//body[@name='{root_name}']")
    root_joint = root.find(".//joint[@type='free']")
    root_body.remove(root_joint)
    root_body.set("pos", "0 0 0")

    for extend_config in cfg.extend_config:
        parent = root.find(f".//body[@name='{extend_config.parent_name}']")
        if parent is None:
            raise ValueError(f"Parent body {extend_config.parent_name} not found in MJCF")
        
        pos = extend_config.pos
        rot = extend_config.rot
        # create and insert a dummy body with a fixed joint
        body = ET.Element("body", name=extend_config.joint_name)
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        body.set("quat", f"{rot[0]} {rot[1]} {rot[2]} {rot[3]}")
        inertial = ET.Element("inertial", pos="0 0 0", quat="0 0 0 1", mass="0.1", diaginertia="0.1 0.1 0.1")
        body.append(inertial)
        parent.append(body)

    cwd = os.getcwd()
    os.chdir(os.path.dirname(mjcf_path))
    chain = pk.build_chain_from_mjcf(ET.tostring(root, method="xml"), body=root_name)
    os.chdir(cwd)
    return chain


@hydra.main(config_path=CONDIF_PATH, config_name="config", version_base=None)
def main(cfg):
    chain = build_chain(cfg.robot)
    chain.forward_kinematics(torch.zeros(1, chain.n_joints))

    motion_path = os.path.join(DATA_PATH, "motion", "0005_Walking001_poses.npz")

    with open(motion_path, "rb") as f:
        motion = dict(np.load(f))
    
    T = motion["poses"].shape[0]
    motion["poses"] = motion["poses"][:, :66].reshape(T, 22, 3)
    body_pose = torch.as_tensor(motion["poses"][:, 1:], dtype=torch.float32)
    hand_pose = torch.zeros(T, 2, 3)
    data = {
        "body_pose": torch.cat([body_pose, hand_pose], dim=1),
        "global_orient": torch.as_tensor(motion["poses"][:, 0], dtype=torch.float32),
        "trans": torch.as_tensor(motion["trans"], dtype=torch.float32),
    }

    body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")

    path = os.path.join(os.path.dirname(__file__), f"{cfg.robot.humanoid_type}_shape.pt")
    fitted_shape = torch.load(path)

    with torch.no_grad():
        result = body_model.forward(
            fitted_shape["betas"],
            body_pose=data["body_pose"].reshape(T, 69),
            global_orient=data["global_orient"],
            transl=data["trans"]
        )
    
    # which joints to match
    robot_body_names = []
    smpl_joint_idx = []
    for robot_body_name, smpl_joint_name in cfg.robot.joint_matches:
        robot_body_names.append(robot_body_name)
        smpl_joint_idx.append(SMPL_BONE_ORDER_NAMES.index(smpl_joint_name))

    # since the betas are changed and so are the SMPL body morphology,
    # we need to make some corrections to avoid ground pentration
    ground_offset = result.vertices[:, :, 2].min()
    smpl_keypoints = result.joints[:, smpl_joint_idx] - ground_offset

    robot_rot = sRot.from_rotvec(data["global_orient"]) * sRot.from_euler("xyz", [np.pi/2, 0., np.pi/2]).inv()
    robot_rotmat = torch.as_tensor(robot_rot.as_matrix(), dtype=torch.float32)

    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints))        
    robot_trans = torch.nn.Parameter(data["trans"].clone() - ground_offset)
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)

    indices = chain.get_all_frame_indices()
    
    def get_robot_keypoints(th: torch.Tensor, trans: torch.Tensor):
        body_pos = chain.forward_kinematics(th, indices) # in robot's root frame
        robot_keypoints = torch.stack([
            body_pos[name].get_matrix()[:, :3, 3]
            for name in robot_body_names
        ], dim=1)
        # convert to world frame
        robot_keypoints = robot_rotmat.unsqueeze(1) @ robot_keypoints.unsqueeze(-1)
        robot_keypoints = robot_keypoints.squeeze(-1) + trans.unsqueeze(1)
        return robot_keypoints
        
    for i in range(200):
        robot_keypoints = get_robot_keypoints(robot_th, robot_trans)
        loss = nn.functional.mse_loss(robot_keypoints, smpl_keypoints)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"iter {i}, loss {100 * loss.item():.3f}")
    
    with torch.no_grad():
        robot_keypoints = get_robot_keypoints(robot_th, robot_trans)
    
    motion_name = motion_path.split("/")[-1].split(".")[0]
    save_path = f"{motion_name}.pt"
    data = {
        "joint_pos": robot_th.data.numpy(),
        "keypoint_pos_w": robot_keypoints.data.numpy(),
        "root_pos_w": robot_trans.data.numpy(),
        "root_quat_w": robot_rot.as_quat(scalar_first=True),
    }
    print(f"Saving to {save_path}")
    np.savez_compressed(save_path, **data)

    def init_mesh(vertices, color=[0.3, 0.3, 0.3]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh = init_mesh(result.vertices[0])
    vis.add_geometry(mesh)

    plane = o3d.geometry.TriangleMesh.create_box(4., 4., 0.01)
    plane.translate([-2., -2., -0.005])
    plane.compute_vertex_normals()
    vis.add_geometry(plane)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.compute_vertex_normals()
    vis.add_geometry(frame)

    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_keypoints[0])
    robot_pcd.colors = o3d.utility.Vector3dVector(torch.tensor([0, 0, 1]).expand_as(robot_keypoints[0]))
    vis.add_geometry(robot_pcd)

    smpl_pcd = o3d.geometry.PointCloud()
    smpl_pcd.points = o3d.utility.Vector3dVector(result.joints[0, smpl_joint_idx])
    smpl_pcd.colors = o3d.utility.Vector3dVector(torch.tensor([1, 0, 0]).expand_as(result.joints[0, smpl_joint_idx]))
    vis.add_geometry(smpl_pcd)

    opt = vis.get_render_option()
    # Set the point size
    point_size = 10.0  # You can adjust this value according to your needs
    opt.point_size = point_size

    for t in range(T):
        mesh.vertices = o3d.utility.Vector3dVector(result.vertices[t]-ground_offset)
        # mesh.compute_vertex_normals()
        # vis.update_geometry(mesh)

        robot_pcd.points = o3d.utility.Vector3dVector(robot_keypoints[t])
        vis.update_geometry(robot_pcd)

        smpl_pcd.points = o3d.utility.Vector3dVector(smpl_keypoints[t])
        vis.update_geometry(smpl_pcd)

        vis.poll_events()
        vis.update_renderer()
    


if __name__ == "__main__":
    main()