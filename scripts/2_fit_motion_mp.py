import os
import torch
import torch.nn as nn
import hydra
import numpy as np
import glob
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
import multiprocessing as mp

from smplx import SMPL
from scipy.spatial.transform import Rotation as sRot, Slerp

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
    root_body.set("pos", "0 0 0")
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


def lerp(x, xp, fp):
    return np.stack([np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])], axis=1)


def slerp(x, xp, fp):
    s = Slerp(xp, sRot.from_rotvec(fp))
    return s(x).as_rotvec()


def fit_motion(cfg, motion_path: str):
    
    with open(motion_path, "rb") as f:
        motion = dict(np.load(f))
    
    fps = int(motion["mocap_framerate"].item())
    T = motion["poses"].shape[0]
    motion["poses"] = motion["poses"][:, :66].reshape(T, 22, 3)
    if fps != int(cfg.target_fps):
        end_t =  motion["poses"].shape[0] / fps
        xp = np.arange(0, end_t, 1 / fps)
        x = np.arange(0, end_t, 1 / int(cfg.target_fps))
        if x[-1] > xp[-1]:
            x = x[:-1]
        motion["poses"] = np.stack([
            slerp(x, xp, motion["poses"][:, i])
            for i in range(22)
        ], axis=1)
        motion["trans"] = lerp(x, xp, motion["trans"])
    
    print(f"Retargeting motion at {motion_path} from {fps} to {cfg.target_fps}")

    chain = build_chain(cfg.robot)
    chain.forward_kinematics(torch.zeros(1, chain.n_joints))

    T = motion["poses"].shape[0]
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
    smpl_keypoints_w = result.joints[:, smpl_joint_idx] - ground_offset

    # again, convert between Y-up and Z-up
    robot_rot = sRot.from_rotvec(data["global_orient"]) * sRot.from_euler("xyz", [np.pi/2, 0., np.pi/2]).inv()
    robot_rotmat = torch.as_tensor(robot_rot.as_matrix(), dtype=torch.float32)

    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints))        
    robot_trans = torch.nn.Parameter(data["trans"].clone() - ground_offset)
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)

    indices = chain.get_all_frame_indices()
    
    def mat_rotate(rotmat, v):
        return (rotmat @ v.unsqueeze(-1)).squeeze(-1)
        
    for i in range(200):
        fk_output = chain.forward_kinematics(robot_th, indices) # in robot's root frame
        robot_keypoints_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, 3]
            for name in robot_body_names
        ], dim=1)        
        # convert to world frame
        robot_keypoints_w = robot_trans.unsqueeze(1) + mat_rotate(robot_rotmat.unsqueeze(1), robot_keypoints_b)

        loss = nn.functional.mse_loss(robot_keypoints_w, smpl_keypoints_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        robot_keypoints_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, 3]
            for name in robot_body_names
        ], dim=1)        
        # convert to world frame
        robot_keypoints_w = robot_trans.unsqueeze(1) + mat_rotate(robot_rotmat.unsqueeze(1), robot_keypoints_b)

    motion_name = motion_path.split("/")[-1].split(".")[0]
    save_path = f"{motion_name}.npz"
    data = {
        "fps": int(cfg.target_fps),
        "joint_pos": robot_th.data.numpy(),
        "root_pos_w": robot_trans.data.numpy(),
        "root_quat_w": robot_rot.as_quat(scalar_first=True),
        "body_pos_w": robot_keypoints_w.data.numpy(),
        "body_pos_b": robot_keypoints_b.data.numpy(),
    }
    np.savez_compressed(save_path, **data)
    return save_path


@hydra.main(config_path=CONDIF_PATH, config_name="config", version_base=None)
def main(cfg):
    if os.path.isdir(cfg.motion_path):
        motion_paths = glob.glob(os.path.join(cfg.motion_path, "**/*.npz"), recursive=True)
    else:
        motion_paths = [cfg.motion_path]

    motion_paths = [path for path in motion_paths if "Walk" in path]
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