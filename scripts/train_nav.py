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
from go1_gym_learn.ppo_nav import Runner

from tqdm import tqdm

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train_nav(headless=True):
    # dirs = glob.glob(f"../runs/{label}/*")
    # logdir = sorted(dirs)[0]
    curr_file_path = os.path.dirname(os.path.abspath(__file__))
    label_locomotion = os.path.join(curr_file_path,"../runs/gait-conditioned-agility/pretrain-v0/train/025417.456545")
    
    print(label_locomotion)

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
    Cfg.env.num_envs = 2
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    Cfg.env.record_video = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.commands.limit_vel_x = [-0.5,0.5]
    Cfg.commands.limit_vel_y = [-0.5,0.5]
    Cfg.commands.limit_vel_yaw = [-0.5,0.5]

    Cfg.asset.penalize_contacts_on = ["thigh","calf","hip","base"]
    Cfg.env.add_box = True

    from go1_gym.envs.wrappers.history_wrapper_nav import HistoryWrapper

    #training
    gpu_id = 0  
    #env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = World(sim_device=f'cuda:{gpu_id}',headless=headless, cfg=Cfg, locomtion_model_dir= label_locomotion)
    runner = Runner(env, device=f"cuda:{gpu_id}")
    logger.log_params(Cfg=vars(Cfg))
    runner.learn(num_learning_iterations=30000, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'object-avoidance-navigation/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    train_nav(headless=True)