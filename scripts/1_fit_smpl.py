import os
import torch
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
    root_body = root.find(".//body")
    root_joint = root.find(".//joint[@type='free']")
    root_body.remove(root_joint)
    root_name = root_body.get("name")

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
    robot_body_names = chain.get_link_names()

    th = torch.zeros([1, chain.n_joints])
    robot_rest_pose = chain.forward_kinematics(th)

    body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")

    # which joints to match
    robot_joint_idx = []
    smpl_joint_idx = []
    fit_target = []
    for robot_body_name, smpl_joint_name in cfg.robot.joint_matches:
        robot_joint_idx.append(robot_body_names.index(robot_body_name))
        smpl_joint_idx.append(SMPL_BONE_ORDER_NAMES.index(smpl_joint_name))
        fit_target.append(robot_rest_pose[robot_body_name].get_matrix()[:, :3, 3])
    
    root_translation = robot_rest_pose[robot_body_names[0]].get_matrix()[:, :3, 3]
    fit_target = torch.stack(fit_target, dim=1)
    fit_target = fit_target - root_translation.unsqueeze(1)
    print(fit_target)

    transl_0 = torch.zeros([1, 3])
    pose_0 = torch.zeros([1, 24, 3])
    
    # align frame convention: SMPL is Y-up, but the robot is Z-up
    pose_0[:, 0] = torch.as_tensor(sRot.from_euler("xyz", [np.pi/2, 0., np.pi/2]).as_rotvec())
    # or equivalently
    # pose_0[:, 0] = torch.as_tensor(sRot.from_quat([.5, .5, .5, .5]).as_rotvec())
    
    # align SMPL's rest pose with the robot
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
    fitted_history = []

    ITERS = 200
    for i in range(ITERS):
        result = body_model.forward(
            betas,
            body_pose=pose_0[:, 1:].reshape(1, 69),
            global_orient=pose_0[:, 0],
            transl=transl_0
        )
        pelvis = result.joints[:, None, 0]
        fitted = (result.joints[:, smpl_joint_idx] - pelvis) * scale
        
        loss = (fit_target - fitted).square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 0 or i == ITERS - 1:
            with torch.no_grad():
                v = (result.vertices[0] - result.joints[:, 0]) * scale
            vertices.append(v)
            fitted_history.append(fitted[0].detach())
            print(f"iter {i}, loss: {loss.item():.3f}")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.compute_vertex_normals()
    vis.add_geometry(frame)

    # Get the render option
    for i, v in enumerate(vertices):
        offset = torch.tensor([0., i, 0.])
        v = v + offset + torch.tensor([2., 0., 0.])
        mesh = init_mesh(v)
        vis.add_geometry(mesh)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fitted_history[i] + offset + torch.tensor([1., 0., 0.]))
        pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(fitted_history[i]))])
        vis.add_geometry(pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fit_target[0] + offset)
        pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(fit_target[0]))])
        vis.add_geometry(pcd)
    
    
    path = os.path.join(os.path.dirname(__file__), f"{cfg.robot.humanoid_type}_shape.pt")
    torch.save({"betas": betas.data, "scale": scale.data}, path)

    opt = vis.get_render_option()
    # Set the point size
    point_size = 10.0  # You can adjust this value according to your needs
    opt.point_size = point_size
    
    vis.run()


if __name__ == "__main__":
    main()