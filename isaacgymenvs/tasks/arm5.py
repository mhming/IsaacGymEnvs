import numpy as np
import os
import torch

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


class Arm5(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

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
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            arm5_asset_file = self.cfg["env"]["asset"].get("assetFileNameArm5", arm5_asset_file)
            cabinet_asset_file = self.cfg["env"]["asset"].get("assetFileNameCabinet", cabinet_asset_file)

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

        # load cabinet asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        arm5_dof_stiffness = to_torch([400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        arm5_dof_damping = to_torch([80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        self.num_arm5_bodies = self.gym.get_asset_rigid_body_count(arm5_asset)
        self.num_arm5_dofs = self.gym.get_asset_dof_count(arm5_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num arm5 bodies: ", self.num_arm5_bodies)
        print("num arm5 dofs: ", self.num_arm5_dofs)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

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

        # set cabinet dof properties
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        # create prop assets
        box_opts = gymapi.AssetOptions()
        box_opts.density = 400
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        arm5_start_pose = gymapi.Transform()
        arm5_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        arm5_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*get_axis_params(0.4, self.up_axis_idx))

        # compute aggregate size
        num_arm5_bodies = self.gym.get_asset_rigid_body_count(arm5_asset)
        num_arm5_shapes = self.gym.get_asset_rigid_shape_count(arm5_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_arm5_bodies + num_cabinet_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_arm5_shapes + num_cabinet_shapes + self.num_props * num_prop_shapes

        self.arm5s = []
        self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            arm5_actor = self.gym.create_actor(env_ptr, arm5_asset, arm5_start_pose, "arm5", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, arm5_actor, arm5_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            cabinet_pose = cabinet_start_pose
            cabinet_pose.p.x += self.start_position_noise * (np.random.rand() - 0.5)
            dz = 0.5 * np.random.rand()
            dy = np.random.rand() - 0.5
            cabinet_pose.p.y += self.start_position_noise * dy
            cabinet_pose.p.z += self.start_position_noise * dz
            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
                drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
                drawer_pose = self.gym.get_rigid_transform(env_ptr, drawer_handle)

                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()
                        prop_state_pose.p.x = drawer_pose.p.x + propx
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = drawer_pose.p.y + propy
                        prop_state_pose.p.z = drawer_pose.p.z + propz
                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1)
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arm5s.append(arm5_actor)
            self.cabinets.append(cabinet_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, arm5_actor, "panda_link7")
        self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, arm5_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, arm5_actor, "panda_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1