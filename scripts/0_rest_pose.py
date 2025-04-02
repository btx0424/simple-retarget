import mujoco
import mujoco.viewer
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mjcf', type=str)
    args = parser.parse_args()

    mjcf = args.mjcf
    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)

    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)

