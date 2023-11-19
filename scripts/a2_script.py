import isaacgym

assert isaacgym
import torch
import numpy as np
import os
import cv2
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.go1.world import World

from tqdm import tqdm

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        action = body.forward(obs["obs_history"])
        return action

    return policy


def saveToVideo(frames, output_video_name, frame_rate=25.0, codec='mp4v'):
    # Get the shape of the first frame to determine video dimensions
    frame_height, frame_width, _ = frames[0].shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_name, fourcc, frame_rate, (frame_width, frame_height))

    # Loop through the frames and write each frame to the video
    for frame in frames:
        # Convert from RGB to BGR (OpenCV uses BGR format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release the VideoWriter
    out.release()

def load_env(label, headless=False):
    # dirs = glob.glob(f"../runs/{label}/*")
    # logdir = sorted(dirs)[0]
    logdir = os.path.join(os.path.abspath(os.path.dirname(__file__)),f"../runs/{label}")

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    Cfg.env.record_video = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = World(sim_device='cuda:0', headless=headless, cfg=Cfg, locomtion_model_dir=logdir)
    # env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic
    policy = load_policy(logdir)
    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    if not os.path.exists("./imdump"):
        os.mkdir("./imdump")

    label = "gait-conditioned-agility/pretrain-v0/train/025417.456545"

    env, policy = load_env(label, headless=headless)
    env.start_recording()

    obs = env.reset()
    for _ in range(100):
        # with torch.no_grad():
        #     action = policy(obs)
        obs, rew, done, info = env.step(torch.Tensor([0,0,0]))

    
    saveToVideo(env.video_frames, "./imdump/test_video.mp4")


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
