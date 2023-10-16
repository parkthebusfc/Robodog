import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

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

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy

def dog_walk(env,policy, obs, num_eval_steps, x_vel_cmd, y_vel_cmd, yaw_vel_cmd):
    # num_eval_steps = 250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    # x_vel_cmd, y_vel_cmd, yaw_vel_cmed = 0.4, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_vels = np.zeros((num_eval_steps,3))

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)
        measured_vels[i,:] = env.base_lin_vel[:, 0].cpu()
    
    return measured_vels

def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # os.mkdir("./imdump")

    label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label, headless=headless)

    env.start_recording()

    obs = env.reset()

    observed_vels = None
    command_vels_array = None
    steps_each = 50
    command_vels = [[0.4,0,0],[0,0.3,0],[-0.3,0,0],[0,-0.5,0],[0,0,0.25]]
    num_eval_steps = 0
    for cmd in command_vels:
        obs_vel = dog_walk(env,policy, obs, steps_each, *cmd)
        num_eval_steps += steps_each
        if observed_vels is None:
            observed_vels = obs_vel
            command_vels_array = np.tile(np.array(cmd),(steps_each,1))
        else:
            observed_vels = np.concatenate((observed_vels, obs_vel), axis = 0)
            command_vels_array = np.concatenate((command_vels_array,np.tile(np.array(cmd),(steps_each,1))), axis=0)
    
    from matplotlib import pyplot as plt
    # for i, x in enumerate(env.video_frames):
    #     plt.imsave("./imdump/frame-"+str(i)+".jpg",x)
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), observed_vels[:,0], color='black', linestyle="-", label="Measured x-velocity")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), command_vels_array[:,0], color='black', linestyle="--", label="Desired X")
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), observed_vels[:,1], color='black', linestyle="-", label="Measured y-velocity")
    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), command_vels_array[:,1], color='black', linestyle="--", label="Desired Y")
    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), observed_vels[:,2], color='black', linestyle="-", label="Measured yaw-velocity")
    axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), command_vels_array[:,2], color='black', linestyle="--", label="Desired Yaw")
    # axs[0].legend()
    # axs[0].set_title("Forward Linear Velocity")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Velocity (m/s)")

    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=True)
