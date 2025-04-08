import os
import torch
import hydra
import json
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
    chain.print_tree()

    # compute the rest-pose body transforms in the root (base link)'s frame
    th = torch.zeros([1, chain.n_joints])
    robot_body_pose = chain.forward_kinematics(th) 

    body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")

    # which (robot body, SMPL joint) paris to match
    robot_body_names = []
    smpl_joint_idx = []
    for robot_body_name, smpl_joint_name in cfg.robot.joint_matches:
        robot_body_names.append(robot_body_name)
        smpl_joint_idx.append(SMPL_BONE_ORDER_NAMES.index(smpl_joint_name))

    root_translation = robot_body_pose[chain.get_link_names()[0]].get_matrix()[:, :3, 3]
    robot_keypoints = [robot_body_pose[name].get_matrix()[:, :3, 3] for name in robot_body_names]
    robot_keypoints = torch.stack(robot_keypoints, dim=1)
    robot_keypoints = robot_keypoints - root_translation.unsqueeze(1)
    print(robot_keypoints)

    # SMPL rest pose
    transl_0 = torch.zeros([1, 3])
    pose_0 = torch.zeros([1, 24, 3])
    # align frame convention: SMPL is Y-up, but the robot is Z-up
    pose_0[:, 0] = torch.as_tensor(sRot.from_euler("xyz", [np.pi/2, 0., np.pi/2]).as_rotvec())
    # or equivalently
    # pose_0[:, 0] = torch.as_tensor(sRot.from_quat([.5, .5, .5, .5]).as_rotvec())
    
    # align SMPL's rest pose with the robot
    # for example, bend elbows in tha cases of Unitree G1 and H1
    for key, value in cfg.robot.smpl_pose_modifier.items():
        rotvec = sRot.from_euler("xyz", eval(value)).as_rotvec()
        pose_0[:, SMPL_BONE_ORDER_NAMES.index(key)] = torch.as_tensor(rotvec)

    # setup the optimization problem
    betas = torch.nn.Parameter(torch.zeros([1, 10]))
    scale = torch.nn.Parameter(torch.ones([3])) # not really necessary
    opt = torch.optim.Adam([betas], lr=0.05)
    
    import open3d as o3d

    def init_mesh(vertices, color=[0.3, 0.3, 0.3]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    
    vertices = []
    smpl_keypoints_history = []

    ITERS = 300
    for i in range(ITERS):
        result = body_model.forward(
            betas,
            body_pose=pose_0[:, 1:].reshape(1, 69),
            global_orient=pose_0[:, 0],
            transl=transl_0
        )
        pelvis = result.joints[:, None, 0]
        smpl_keypoints = (result.joints[:, smpl_joint_idx] - pelvis) * scale
        
        loss = (robot_keypoints - smpl_keypoints).square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 0 or i == ITERS - 1:
            with torch.no_grad():
                v = (result.vertices[0] - result.joints[:, 0]) * scale
            vertices.append(v)
            smpl_keypoints_history.append(smpl_keypoints[0].detach())
            print(f"iter {i}, loss: {loss.item():.3f}")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.compute_vertex_normals()
    vis.add_geometry(frame)

    for i, v in enumerate(vertices):
        offset = torch.tensor([0., i, 0.])
        v = v + offset + torch.tensor([2., 0., 0.])
        mesh = init_mesh(v)
        vis.add_geometry(mesh)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(smpl_keypoints_history[i] + offset + torch.tensor([1., 0., 0.]))
        pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(smpl_keypoints_history[i]))])
        vis.add_geometry(pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(robot_keypoints[0] + offset)
        pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(robot_keypoints[0]))])
        vis.add_geometry(pcd)
    
    
    meta = {
        "body_names": chain.get_link_names(),
        "joint_names": [joint.name for joint in chain.get_joints()]
    }
    with open("meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    path = os.path.join(os.path.dirname(__file__), f"{cfg.robot.humanoid_type}_shape.pt")
    torch.save({"betas": betas.data, "scale": scale.data}, path)

    # increase point size for better visualization
    opt = vis.get_render_option()
    point_size = 10.0  # You can adjust this value according to your needs
    opt.point_size = point_size
    
    vis.run()


if __name__ == "__main__":
    main()