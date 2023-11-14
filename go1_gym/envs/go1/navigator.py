from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from isaacgym import gymtorch, gymapi, gymutil
import torch
from params_proto import Meta
from typing import Union
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.utils.terrain import Terrain
import os
from isaacgym.torch_utils import *
import math

from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.base.legged_robot_config import Cfg

class Navigator(VelocityTrackingEasyEnv):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                    cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX", locomtion_model_dir = "gait-conditioned-agility/pretrain-v0/train/025417.456545", behavior_commands = {
                        "body_height" : 0.0,
                        "step_frequency" : 3.0,
                        "gait" : torch.Tensor([0.5, 0, 0]),
                        "durations" : 0.5,
                        "footswing_height" : 0.08,
                        "pitch" : 0.0,
                        "roll" : 0.0,
                        "stance_width" : 0.25
                    }):
        self.locomotion_model_dir = locomtion_model_dir
        self.behavior = behavior_commands
        self.load_locomotion_policy()
        super().__init__(sim_device, headless, num_envs, prone,deploy,cfg, eval_cfg,initial_dynamics_dict,physics_engine)
        
        self.obs_history_length = self.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, dtype=torch.float,
                                        device=self.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 3] = self.behavior["body_height"]
        self.commands[env_ids, 4] = self.behavior["step_frequency"]
        self.commands[env_ids, 5:8] =self.behavior["gait"]
        self.commands[env_ids, 8] = self.behavior["durations"]
        self.commands[env_ids, 9] = self.behavior["footswing_height"]
        self.commands[env_ids, 10] = self.behavior["pitch"]
        self.commands[env_ids, 11] = self.behavior["roll"]
        self.commands[env_ids, 12] = self.behavior["stance_width"]
        
    def load_locomotion_policy(self):
        body = torch.jit.load(self.locomotion_model_dir + '/checkpoints/body_latest.jit')
        import os
        adaptation_module = torch.jit.load(self.locomotion_model_dir + '/checkpoints/adaptation_module_latest.jit')

        def policy(obs, info={}):
            i = 0
            latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
            action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
            info['latent'] = latent
            return action

        self.locomotion_policy = policy
    
    def velocity_step(self, actions):
        return super().step(actions=actions)

    def get_observation_buffer(self):
        return self.obs_buf

    def get_observations(self):
        obs = self.get_observation_buffer()
        privileged_obs = self.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def step(self, cmd):
        # cmds are the velocities that nav policy outputs
        # Our step function takes these cmds and passes it through the locomotion policy, applies the computed torques, computes observations, rewards and so on
        self.commands[:, :3] = cmd
        
        obs = self.get_observations()
        locomotion_actions = self.locomotion_policy(obs)
        obs, rew, done, info = self.velocity_step(locomotion_actions)
        privileged_obs = info["privileged_obs"]
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim=-1)

        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew, done, info

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret
    
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
