import os
import torch
import mujoco
import mujoco.viewer
import hydra
import time

CONDIF_PATH = os.path.join(os.path.dirname(__file__), "../cfg")

@hydra.main(config_path=CONDIF_PATH, config_name="config", version_base=None)
def main(cfg):
    mjcf_path = cfg.robot.mjcf_path

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    motion = torch.load("retargeted.pt")

    viewer = mujoco.viewer.launch_passive(model, data)
    t = 0
    while viewer.is_running():
        data.qpos[7:] = motion["joint_pos"][t].numpy()
        data.qpos[:3] = motion["root_pos"][t].numpy()
        data.qpos[3:7] = motion["root_quat"][t].numpy()
        mujoco.mj_forward(model, data)
        viewer.sync()
        t += 1
        time.sleep(1 / motion["fps"])


if __name__ == "__main__":
    main()