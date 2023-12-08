import isaacgym

assert isaacgym
import torch
import numpy as np
import cv2
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.go1.world import World
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

def load_policy_nav(logdir, device="cpu"):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit', map_location=device)
    import os

    def policy_nav(obs, info={}):
        i = 0
        action = body.forward(obs["nav_obs_history"].to('cuda:0'))
        return action

    return policy_nav

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

def load_env(label_nav, label_locomotion, headless=False):

    with open(label_locomotion + "/parameters.pkl", 'rb') as file:
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
    Cfg.env.add_box = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper_nav import HistoryWrapper

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = World(sim_device=device,headless=headless, cfg=Cfg, locomtion_model_dir=label_locomotion)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_nav.actor_critic_nav import ActorCritic_Nav

    policy_nav = load_policy_nav(label_nav, device)
    return env, policy_nav

def dog_walk(env, policy_nav, obs, num_eval_steps = 750):
    # num_eval_steps = 250

    measured_vels = np.zeros((num_eval_steps,3))
    i = 0
    done = False
    while not done and i < num_eval_steps:
        with torch.no_grad():
            actions_nav = policy_nav(obs)
        
        obs, rew, done, info = env.step(actions_nav)
        measured_vels[i,:] = env.base_lin_vel[0, :].cpu()
        measured_vels[i,2] = env.base_ang_vel[0, 2].cpu()
        i+=1
    return measured_vels

def play_go1(headless=False):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    curr_file_path = os.path.dirname(os.path.abspath(__file__))
    label_nav = os.path.join(curr_file_path,"ak2227/scratch/2023/11-27/155016")

    label_locomotion = os.path.join(curr_file_path,"../runs/gait-conditioned-agility/2023-11-03/train/210513.245978")
    
    env, policy_nav = load_env(label_nav, label_locomotion, headless=headless)

    env.start_recording()

    obs = env.reset()

    dog_walk(env, policy_nav, obs)
    
    write_video(env.complete_video_frames, "./video.avi")


def write_video(images, video_path, fps=30):
  images = np.array(images)
  height, width, channels = images.shape[1:4]

  fourcc = cv2.VideoWriter_fourcc(*'DIVX')

  writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

  for image in images:
    writer.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  writer.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=True)