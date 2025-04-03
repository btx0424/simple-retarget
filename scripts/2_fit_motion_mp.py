import os
import torch
import torch.nn as nn
import hydra
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
import glob
import multiprocessing as mp
import pathlib

from smplx import SMPL
from scipy.spatial.transform import Rotation as sRot

OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
if OMP_NUM_THREADS <= 1:
    raise ValueError("Set OMP_NUM_THREADS > 1 to enable multiprocessing. Current value: %s" % OMP_NUM_THREADS)

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

def fit_motion(cfg, motion_path: str):
    with open(motion_path, "rb") as f:
        raw = dict(np.load(f))
    print(motion_path)
    
    chain = build_chain(cfg.robot)
    chain.forward_kinematics(torch.zeros(1, chain.n_joints))

    T = raw["poses"].shape[0]
    body_pose = raw["poses"][:, :66]
    body_pose = torch.as_tensor(body_pose, dtype=torch.float32)
    hand_pose = torch.zeros(T, 6)
    data = {
        "pose": torch.cat([body_pose, hand_pose], dim=1).reshape(T, 24, 3),
        "trans": torch.as_tensor(raw["trans"], dtype=torch.float32),
        "gender": raw["gender"].item(),
        "betas": torch.as_tensor(raw["betas"], dtype=torch.float32),
        "fps": raw["mocap_framerate"].item()
    }
    body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")

    path = os.path.join(os.path.dirname(__file__), f"{cfg.robot.humanoid_type}_shape.pt")
    fitted_shape = torch.load(path)

    with torch.no_grad():
        result = body_model.forward(
            fitted_shape["betas"],
            # data["betas"][:body_model.num_betas].unsqueeze(0),
            body_pose=data["pose"][:, 1:].reshape(T, 69),
            global_orient=data["pose"][:, 0].reshape(T, 3),
            transl=data["trans"].reshape(T, 3)
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

    robot_rot = sRot.from_rotvec(data["pose"][:, 0]) * sRot.from_euler("xyz", [np.pi/2, 0., np.pi/2]).inv()
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
        root_translation = body_pos["pelvis"].get_matrix()[:, :3, 3].unsqueeze(1)
        # convert to world frame
        robot_keypoints = robot_rotmat.unsqueeze(1) @ (robot_keypoints - root_translation).unsqueeze(-1)
        robot_keypoints = robot_keypoints.squeeze(-1) + trans.unsqueeze(1)
        return robot_keypoints
        
    for i in range(200):
        robot_keypoints = get_robot_keypoints(robot_th, robot_trans)
        loss = nn.functional.mse_loss(robot_keypoints, smpl_keypoints)
        opt.zero_grad()
        loss.backward()
        opt.step()
    robot_keypoints = robot_keypoints.detach()
    
    motion_name = motion_path.split("/")[-1].split(".")[0]
    save_path = f"{motion_name}.pt"
    torch.save({
        "fps": data["fps"],
        "joint_pos": robot_th.data,
        "root_pos": robot_trans.data,
        "root_quat": torch.as_tensor(robot_rot.as_quat(scalar_first=True)),
    }, save_path)
    return save_path


@hydra.main(config_path=CONDIF_PATH, config_name="config", version_base=None)
def main(cfg):
    if os.path.isdir(cfg.motion_path):
        motion_paths = glob.glob(os.path.join(cfg.motion_path, "**/*.npz"), recursive=True)
    else:
        motion_paths = [cfg.motion_path]

    print(f"Found {len(motion_paths)} motion files under {cfg.motion_path}")

    from tqdm import tqdm
    with mp.Pool(processes=4) as pool, tqdm(total=len(motion_paths)) as pbar:
        results = [pool.apply_async(fit_motion, (cfg, motion_path)) for motion_path in motion_paths]
        for result in results:
            save_path = result.get()
            print(f"Saved to {save_path}")
            pbar.update(1)
            pbar.refresh()

if __name__ == "__main__":
    main()