from go1_gym.envs.go1.navigator import Navigator
from isaacgym import gymtorch, gymapi, gymutil
import torch
from params_proto import Meta
from typing import Union
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.utils.terrain import Terrain
import os
from isaacgym.torch_utils import *
import math

from go1_gym.envs.base.legged_robot_config import Cfg

class World(Navigator):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX", locomtion_model_dir = "gait-conditioned-agility/pretrain-v0/train/025417.456545"):
        super().__init__(sim_device, headless, num_envs, prone,deploy,cfg,eval_cfg,initial_dynamics_dict,physics_engine, locomtion_model_dir=locomtion_model_dir)

        self.num_actions = 3

    def update_goals(self, env_ids):
        self.init_root_states = self.root_states[self.num_actors_per_env * env_ids, :3]
        self.goals = self.root_states[self.num_actors_per_env * env_ids, :3]
        self.goals[:, 0:1] += 3 * torch.ones((len(env_ids), 1)).to(self.goals.device)
        print("Computed Goals!")

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()


        def make_wall(env_handle, start_pose, dimensions, env_num, wall_id):
            wall_asset_options = gymapi.AssetOptions()
            wall_asset_options.use_mesh_materials = True
            wall_asset_options.disable_gravity = True
            wall_asset_options.fix_base_link = True
            wall_asset = self.gym.create_box(self.sim, *dimensions, wall_asset_options)
            wall_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
            wall_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
            rigid_shape_props = self._process_rigid_shape_props(wall_rigid_shape_props, env_num)
            self.gym.set_asset_rigid_shape_properties(wall_asset, rigid_shape_props)
            wall_handle = self.gym.create_actor(env_handle, wall_asset, start_pose , f"wall_{wall_id}", env_num,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(wall_dof_props, env_num)
            self.gym.set_actor_dof_properties(env_handle, wall_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, wall_handle)
            body_props = self._process_rigid_body_props(body_props, env_num)
            self.gym.set_actor_rigid_body_properties(env_handle, wall_handle, body_props, recomputeInertia=True)
            return wall_handle
        
        self.wall_handles = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            ## Adding Robot
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)

            ## Adding walls
            offset = [
                (1.5,1,1),
                (3.5,0,1),
                (1.5,-1,1),
                (-0.5,0,1),
            ]
            dimensions = [
                (4,0.12,2.0),
                (0.12,2.0,2.0),
                (4.0,0.12,2.0),
                (0.12,2.0,2.0),
            ]
            
            for j in range(4):
                ofs = gymapi.Vec3(*offset[j])
                dim = dimensions[j]

                tmp_pose = gymapi.Transform()
                tmp_pose.p = start_pose.p + ofs 

                wall_handle = make_wall(env_handle, tmp_pose, dim, i,j) 
                self.wall_handles.append(wall_handle)


            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
        
        self.num_actors_per_env = (len(self.actor_handles) + len(self.wall_handles)) // self.num_envs
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 1920
            self.camera_props.height = 1080
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []
    
    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        print("Num Actors: ",self.num_actors_per_env)
        ### Transform of robot which is the first actor of each environment
        # base position 
        if self.custom_origins:
            self.root_states[self.num_actors_per_env * env_ids] = self.base_init_state
            self.root_states[self.num_actors_per_env * env_ids, :3] += self.env_origins[env_ids]
            self.root_states[self.num_actors_per_env * env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
                                                               cfg.terrain.x_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[self.num_actors_per_env * env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
                                                               cfg.terrain.y_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[self.num_actors_per_env * env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[self.num_actors_per_env * env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[self.num_actors_per_env * env_ids] = self.base_init_state
            self.root_states[self.num_actors_per_env * env_ids, :3] += self.env_origins[env_ids]
        
        self.update_goals(env_ids)
        # base yaws
        init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
                                     cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[self.num_actors_per_env * env_ids, 3:7] = quat

        # base velocities
        self.root_states[self.num_actors_per_env * env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = torch.arange(self.root_states.shape[0], dtype=torch.int32, device=env_ids.device)

        
        status = self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        if not status:
            raise SystemError("Could not set necessary transforms")
        if cfg.env.record_video:
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 4.0, bz + 3.0),
                                         gymapi.Vec3(bx, by, bz))

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]
            self.video_frames_eval = []
    
    def compute_reward(self):
        self.rew_buf[:] = 0
        env_ids = torch.arange(self.num_envs)
        robot_pos = self.root_states[self.num_actors_per_env * env_ids,0:3]
        
        # reward structure (+ve main task) * exp(negative auxiliary rewards)

        # going to the wall
        task_reward = torch.exp(-1 * torch.norm(robot_pos[:,0:1] - self.goals[:,0]))

        # auxiliary rewards
        # avoiding timeouts
        timeout_penalty = self.time_out_buf.to(torch.int)

        # avoid walls
        wall_penalty = None
        for wall_num in range(4):
            if wall_penalty is None:
                wall_penalty =  torch.norm(self.root_states[self.num_actors_per_env * env_ids + wall_num + 1,0:3] - self.goals)
            else:
                wall_penalty += torch.norm(self.root_states[self.num_actors_per_env * env_ids + wall_num + 1,0:3] - self.goals)

        self.rew_buf = task_reward * torch.exp(-0.1*(timeout_penalty * 0.01 + wall_penalty * 0.03 ))
        
        # self.rew_buf = (self.root_states[self.num_actors_per_env * env_ids,0:1] - self.goals[:,0:1])[:,0]
        # self.rew_buf -= 2 * torch.abs(self.init_root_states[:,1] - self.root_states[self.num_actors_per_env* env_ids, 1])
        # self.rew_buf += -10 * self.time_out_buf

        # self.episode_sums["goal_reward"] += (self.root_states[self.num_actors_per_env * env_ids,0:1] - self.goals[:,0:1])[:,0]
        # init_root_states_column = self.init_root_states[:, 1].unsqueeze(1)
        # penalty = -2 * torch.abs(init_root_states_column - self.root_states[self.num_actors_per_env * env_ids, :])
        # #self.episode_sums["wall_penalty"] += -2 * torch.abs(self.init_root_states[:,1] - self.root_states[self.num_actors_per_env* env_ids, ])
        # self.episode_sums["wall_penalty"] += penalty.sum()
        # self.episode_sums["timeout_penalty"] += -10 * self.time_out_buf
        # self.episode_sums["total"] += self.rew_buf

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf = self.time_out_buf
        env_ids = torch.arange(self.num_envs)
        self.reset_buf |= (self.root_states[self.num_actors_per_env * env_ids, 0] > self.goals[:, 0])