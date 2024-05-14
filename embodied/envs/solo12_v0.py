import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.75,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class Solo12Env(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
            self,
            xml_file="/home/ubuntu/victor/learning_world_model/gym/envs/assets/scene.xml",
            ctrl_cost_weight=0.5,
            use_contact_forces=False,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True, # default True
            healthy_z_range=(-0.35, 0.5),
            goal_z = -0.2,
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            goal_z,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
            )
        
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._goal_z = goal_z
        self._time_limit = 25
        self._time_elapsed = 0

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 19
        if not self._exclude_current_positions_from_observation:
            obs_shape += 6
        if use_contact_forces:
            obs_shape += 12
            # TODO: check what dimensions are correct for all obs_shapes defined for Solo12
        
        observation_space = Dict(
            {"state":
            Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_shape,),
            dtype=np.float64,
        ),
            "image":
            Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
            ),}
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            camera_id=0,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )
    
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces
    
    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(np.square(self.contact_forces))
        return contact_cost
    
    @property
    def is_healthy(self):
        state = self.get_body_com("base_link")
        min_z, max_z = self._healthy_z_range
        #print(self.get_body_com("base_link")[:3])
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy
    
    @property
    def terminated(self):
        healthy = not self.is_healthy if self._terminate_when_unhealthy else False
        time_up = self._time_elapsed >= self._time_limit
        return healthy or time_up
    
    def step(self, action):
      healthy_reward = self.healthy_reward
      # get velocity 
      xy_position_before = self.get_body_com("base_link")[:2].copy()
      self.do_simulation(action, self.frame_skip)
      xy_position_after = self.get_body_com("base_link")[:2].copy()
      xy_velocity = (xy_position_after - xy_position_before) / self.dt
      self._time_elapsed += self.dt

    
      #print("pos: ", base_link_pos[:2])
      reward = xy_velocity[0] + xy_velocity[1]
      terminated = self.terminated
      observation = self._get_obs()
      info = {
            "reward_survive": healthy_reward,
            "reward": reward,
            "xy_velocity": xy_velocity,
            "is_terminal": terminated,
            "time_elapsed": self._time_elapsed,
        }
      
      if self.render_mode == "human":
          self.render()
      # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
      return observation, reward, terminated, info  
    
    def _get_obs(self):
        obs = {}
        obs["state"] = self.data.qpos.flat.copy()
        obs["image"] = self.render()
        return obs

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            contact_forces = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_forces))
        else:
            return np.concatenate((position, velocity))
        

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
    
    def reset(
        self,
    ):
        self._reset_simulation()
        ob = self.reset_model()
        self._time_elapsed = 0
        return ob
    
    def render(self):
        return super().render()

       
