import numpy as np
import os
import torch

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


class Arm5(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        num_obs = 15
        num_acts = 5

        self.cfg["env"]["numObservations"] = 15
        self.cfg["env"]["numActions"] = 5

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.arm5_default_dof_pos = to_torch([0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm5_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm5_dofs]
        self.arm5_dof_pos = self.arm5_dof_state[..., 0]
        self.arm5_dof_vel = self.arm5_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.arm5_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32,
        #                                    device=self.device).view(self.num_envs, -1)
        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        arm5_asset_file = "urdf/arm5_description/urdf/arm5.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            arm5_asset_file = self.cfg["env"]["asset"].get("assetFileNameArm5", arm5_asset_file)

        # load arm5 asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        arm5_asset = self.gym.load_asset(self.sim, asset_root, arm5_asset_file, asset_options)

        arm5_dof_stiffness = to_torch([400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        arm5_dof_damping = to_torch([80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        self.num_arm5_bodies = self.gym.get_asset_rigid_body_count(arm5_asset)
        self.num_arm5_dofs = self.gym.get_asset_dof_count(arm5_asset)

        print("num arm5 bodies: ", self.num_arm5_bodies)
        print("num arm5 dofs: ", self.num_arm5_dofs)

        # set arm5 dof properties
        arm5_dof_props = self.gym.get_asset_dof_properties(arm5_asset)
        self.arm5_dof_lower_limits = []
        self.arm5_dof_upper_limits = []
        for i in range(self.num_arm5_dofs):
            arm5_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                arm5_dof_props['stiffness'][i] = arm5_dof_stiffness[i]
                arm5_dof_props['damping'][i] = arm5_dof_damping[i]
            else:
                arm5_dof_props['stiffness'][i] = 7000.0
                arm5_dof_props['damping'][i] = 50.0

            self.arm5_dof_lower_limits.append(arm5_dof_props['lower'][i])
            self.arm5_dof_upper_limits.append(arm5_dof_props['upper'][i])

        self.arm5_dof_lower_limits = to_torch(self.arm5_dof_lower_limits, device=self.device)
        self.arm5_dof_upper_limits = to_torch(self.arm5_dof_upper_limits, device=self.device)
        self.arm5_dof_speed_scales = torch.ones_like(self.arm5_dof_lower_limits)
        # self.arm5_dof_speed_scales[[3, 4]] = 0.1# 无手指 不设置
        # arm5_dof_props['effort'][3] = 200
        # arm5_dof_props['effort'][4] = 200

        arm5_start_pose = gymapi.Transform()
        arm5_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        arm5_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # compute aggregate size
        num_arm5_bodies = self.gym.get_asset_rigid_body_count(arm5_asset)
        num_arm5_shapes = self.gym.get_asset_rigid_shape_count(arm5_asset)
        max_agg_bodies = num_arm5_bodies
        max_agg_shapes = num_arm5_shapes

        self.arm5s = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            arm5_actor = self.gym.create_actor(env_ptr, arm5_asset, arm5_start_pose, "arm5", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, arm5_actor, arm5_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arm5s.append(arm5_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, arm5_actor, "link5")

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.arm5s[0], "link5")
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        arm5_local_grasp_pose = hand_pose_inv
        arm5_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.arm5_local_grasp_pos = to_torch([arm5_local_grasp_pose.p.x, arm5_local_grasp_pose.p.y,
                                                arm5_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.arm5_local_grasp_rot = to_torch([arm5_local_grasp_pose.r.x, arm5_local_grasp_pose.r.y,
                                                arm5_local_grasp_pose.r.z, arm5_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        drawer_local_grasp_pose = gymapi.Transform()
        drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
                                                drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
                                                drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.arm5_grasp_pos = torch.zeros_like(self.arm5_local_grasp_pos)
        self.arm5_grasp_rot = torch.zeros_like(self.arm5_local_grasp_rot)
        self.arm5_grasp_rot[..., -1] = 1  # xyzw
        self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        self.drawer_grasp_rot[..., -1] = 1

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_arm5_reward(
            self.reset_buf, self.progress_buf, self.actions, self.cabinet_dof_pos,
            self.arm5_grasp_pos, self.drawer_grasp_pos, self.arm5_grasp_rot, self.drawer_grasp_rot,
            self.arm5_lfinger_pos, self.arm5_rfinger_pos,
            self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )


    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]

        self.arm5_grasp_rot[:], self.arm5_grasp_pos[:], self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.arm5_local_grasp_rot, self.arm5_local_grasp_pos,
                                     drawer_rot, drawer_pos, self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
                                     )

        self.arm5_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.arm5_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.arm5_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.arm5_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.arm5_dof_pos - self.arm5_dof_lower_limits)
                          / (self.arm5_dof_upper_limits - self.arm5_dof_lower_limits) - 1.0)
        to_target = self.drawer_grasp_pos - self.arm5_grasp_pos
        self.obs_buf = torch.cat((dof_pos_scaled, self.arm5_dof_vel * self.dof_vel_scale, to_target,
                                  self.cabinet_dof_pos[:, 3].unsqueeze(-1), self.cabinet_dof_vel[:, 3].unsqueeze(-1)), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset arm5
        pos = tensor_clamp(
            self.arm5_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_arm5_dofs), device=self.device) - 0.5),
            self.arm5_dof_lower_limits, self.arm5_dof_upper_limits)
        self.arm5_dof_pos[env_ids, :] = pos
        self.arm5_dof_vel[env_ids, :] = torch.zeros_like(self.arm5_dof_vel[env_ids])
        self.arm5_dof_targets[env_ids, :self.num_arm5_dofs] = pos

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)
        targets = self.arm5_dof_targets[:, :self.num_arm5_dofs] + self.arm5_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.arm5_dof_targets[:, :self.num_arm5_dofs] = tensor_clamp(
            targets, self.arm5_dof_lower_limits, self.arm5_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.arm5_dof_targets))

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # self.compute_observations()
        # self.compute_reward(self.actions)

    #####################################################################
    ###=========================jit functions=========================###
    #####################################################################

    @torch.jit.script
    def compute_arm5_reward(
            reset_buf, progress_buf, actions, cabinet_dof_pos,
            arm5_grasp_pos, drawer_grasp_pos, arm5_grasp_rot, drawer_grasp_rot,
            arm5_lfinger_pos, arm5_rfinger_pos,
            gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
            num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
            finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

        # distance from hand to the drawer
        d = torch.norm(arm5_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(arm5_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(arm5_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of forward axis for gripper
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(
            -1)  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(arm5_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                           torch.where(arm5_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                       around_handle_reward + 0.5, around_handle_reward),
                                           around_handle_reward)
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(arm5_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(arm5_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(arm5_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                         torch.where(arm5_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                     (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward),
                                         finger_dist_reward)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions ** 2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
                  + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
                  + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        # prevent bad style in opening drawer
        rewards = torch.where(arm5_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                              torch.ones_like(rewards) * -1, rewards)
        rewards = torch.where(arm5_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
                              torch.ones_like(rewards) * -1, rewards)

        # reset if drawer is open or max length reached
        reset_buf = torch.where(cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(reset_buf), reset_buf)
        reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

        return rewards, reset_buf

    @torch.jit.script
    def compute_grasp_transforms(hand_rot, hand_pos, arm5_local_grasp_rot, arm5_local_grasp_pos,
                                 drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                                 ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        global_arm5_rot, global_arm5_pos = tf_combine(
            hand_rot, hand_pos, arm5_local_grasp_rot, arm5_local_grasp_pos)
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

        return global_arm5_rot, global_arm5_pos, global_drawer_rot, global_drawer_pos