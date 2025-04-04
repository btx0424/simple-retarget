import os
import numpy as np
import mujoco
import mujoco.viewer
import hydra
import time
import glob
import threading
from scipy.spatial.transform import Rotation as sRot, Slerp

CONDIF_PATH = os.path.join(os.path.dirname(__file__), "../cfg")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")


class MotionVis:
    def __init__(self, motions):
        self.motions = motions
        self.i = 0
        self.t = 0
        self.motion = self.motions[self.i]

    def key_callback(self, keycode):
        print(keycode)
        if keycode == 265: # up arrow, prev motion
            self.i = (self.i - 1) % len(self.motions)
            self.motion = self.motions[self.i]
            self.t = 0
        elif keycode == 264: # down arrow, next motion
            self.i = (self.i + 1) % len(self.motions)
            self.motion = self.motions[self.i]
            self.t = 0

    def run(self, model, data: mujoco.MjData, viewer: mujoco.viewer.Handle):
        while viewer.is_running():
            with viewer.lock():
                data.qpos[7:] = self.motion["joint_pos"][self.t]
                data.qpos[:3] = self.motion["root_pos_w"][self.t]
                data.qpos[3:7] = self.motion["root_quat_w"][self.t]
                self.t = (self.t + 1) % self.motion["joint_pos"].shape[0]
                mujoco.mj_forward(model, data)
            time.sleep(1 / self.motion["fps"])


def lerp(x, xp, fp):
    return np.stack([np.interp(x, xp, fp[:, i]) for i in range(fp.shape[1])], axis=1)


def slerp(x, xp, fp):
    s = Slerp(xp, sRot.from_quat(fp, scalar_first=True))
    return s(x).as_quat(scalar_first=True)


@hydra.main(config_path=CONDIF_PATH, config_name="config", version_base=None)
def main(cfg):
    mjcf_path = cfg.robot.mjcf_path

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    motion_paths = glob.glob(os.path.join(cfg.motion_path, "*.npz"))
    motions = []
    target_fps = cfg.target_fps
    for motion_path in motion_paths:
        motion = dict(np.load(motion_path))
        if motion["fps"] != target_fps:
            T = motion["joint_pos"].shape[0]
            end_t = T / motion["fps"]
            xp = np.arange(0, end_t, 1 / motion["fps"])
            x = np.arange(0, end_t, 1 / 50)
            if x[-1] > xp[-1]:
                x = x[:-1]
            motion["root_pos_w"] = lerp(x, xp, motion["root_pos_w"])
            motion["joint_pos"] = lerp(x, xp, motion["joint_pos"])
            motion["root_quat_w"] = slerp(x, xp, motion["root_quat_w"])
            motion["fps"] = 50
        motions.append(motion)

    motion_vis = MotionVis(motions)
    
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=motion_vis.key_callback)
    
    def render_async():
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    threading.Thread(target=render_async).start()
    
    motion_vis.run(model, data, viewer)


if __name__ == "__main__":
    main()