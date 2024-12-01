# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import sys
import open3d as o3d
import numpy as np
import pygame
import random
import time
import threading

from datetime import datetime
from matplotlib import cm
from skimage.transform import resize
from PIL import Image

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import carla

# Local module imports
from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
from gym_carla.envs.actor_utils import *
from gym_carla.primary_actors import *
from gym_carla.sensors import CollisionDetector, CameraSensors, LIDARSensor, RadarSensor

class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params, writer=None):
    ...
    self.total_step = 0
    self.display_size = params['display_size']
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    self.writer = writer

    # Action space
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer

    # # Observation space
    # self.observation_space = spaces.Box(
    #   low=0, high=255,
    #   shape=(4, self.obs_size, self.obs_size),
    #   dtype=np.uint8
    # )

    self.observation_space = spaces.Dict({
      'camera': spaces.Box(
        low=0, high=255, 
        shape=(4, self.obs_size, self.obs_size), 
        dtype=np.uint8),
      'state': spaces.Box(
        low=-np.inf, high=np.inf, 
        shape=(4,), 
        dtype=np.float32),
    })

    # Connect to CARLA server and get world object
    print('Connecting to CARLA server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(4000.0)
    self.world = client.load_world(params['town'])
    print('CARLA server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = get_vehicle_spawn_points(self.world)
    print(f"Retrieved {len(self.vehicle_spawn_points)} vehicle spawn points.")

    self.walker_spawn_points = generate_walker_spawn_points(self.world, self.number_of_walkers)
    print(f"Generated {len(self.walker_spawn_points)} valid walker spawn points out of {self.number_of_walkers} requested.")

    # Create the ego vehicle blueprint
    self.ego_bp = create_vehicle_blueprint(self.world, params['ego_vehicle_filter'], color='49,8,8')

    # Initialize sensors
    self.collision_detector = CollisionDetector(self.world)
    self.camera_sensors = CameraSensors(self.world, self.obs_size, self.display_size)
    self.lidar_sensor = LIDARSensor(self.world)
    self.radar_sensor = RadarSensor(self.world)

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    # Initialize the renderer
    self._init_renderer()

  def reset(self, seed = None, options = None):
    # Disable sync mode
    self._set_synchronous_mode(False)

    # Delete sensors, vehicles and walkers
    clear_all_actors(self.world, [
        'sensor.other.collision', 'sensor.camera.rgb',
        'sensor.other.radar', 'sensor.lidar.ray_cast',
        'vehicle.*', 'controller.ai.walker', 'walker.*'
    ])

    # Clear sensor objects
    self.collision_detector.collision_detector = None
    self.camera_sensors.camera_sensors = None
    #self.lidar_sensor.lidar_sensor = None
    #self.radar_sensor.radar_sensor = None

    # Spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    vehicles_spawned = spawn_vehicles(self.world, self.vehicle_spawn_points, self.number_of_vehicles)

    walkers_spawned = spawn_walkers(self.world, self.walker_spawn_points, self.number_of_walkers)
    print(f"Successfully spawned {walkers_spawned} out of {self.number_of_walkers} walkers.")

    # Get vehicle polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = get_actor_polygons(self.world, 'vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)

    # Get walker polygon list
    self.walker_polygons = []
    walker_poly_dict = get_actor_polygons(self.world, 'walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      transform = random.choice(self.vehicle_spawn_points)

      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)

    # Spawn and attach sensors
    self.collision_detector.spawn_and_attach(self.ego)
    self.collision_detector.clear_collision_history()
    self.camera_sensors.spawn_and_attach(self.ego)

    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    info = self._get_info()

    # Set ego information for render
    self.birdeye_render.set_hero(self.ego, self.ego.id)
    return self._get_obs(), info

  def step(self, action):
    # Calculate acceleration and steering
    if self.discrete:
      acc = self.discrete_act[0][action//self.n_steer]
      steer = self.discrete_act[1][action%self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)

    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    # Tick the world
    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = get_actor_polygons(self.world, 'vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = get_actor_polygons(self.world, 'walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # Route planner
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Get reward and log components
    total_reward, reward_components = self._get_reward(self.total_step)
    if self.writer:
        for key, value in reward_components.items():
            self.writer.add_scalar(f"Reward/{key}", value, self.total_step)

    # Check termination conditions
    terminated = self._terminal()

    # TODO episode truncates if last waypoint is reached
    truncated = False

    # Update timesteps
    self.time_step += 1   # Episode timestep
    self.total_step += 1  # Global timestep

    # Prepare info dictionary
    info = self._get_info()
    info["reward_components"] = reward_components  # Include reward components for debugging

    return (self._get_obs(), total_reward, self._terminal(), truncated, info)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.display = pygame.display.set_mode(
    (self.display_size * 6, self.display_size),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=vehicle
      return True

    return False

  def _get_obs(self):
    """Get the observations."""
    # Birdeye rendering
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    #self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    # Birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # Display birdeye image
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    # Display on Pygame (for visualization only)
    self.camera_sensors.display_camera_img(self.display)
    pygame.display.flip()

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi                          # Convert yaw to radians
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y) # Calculate lateral distance and heading error to lane center
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()   
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    # Retrieve optimized camera images for RL
    camera_images = self.camera_sensors.camera_img
    obs = {
      'camera':camera_images,
      'state':state
    }

    return obs

  def _get_reward(self, step):
      """Calculate the reward based on waypoint following and log key information for debugging."""
      # Reward for speed tracking
      v = self.ego.get_velocity()
      speed = np.sqrt(v.x**2 + v.y**2)
      r_speed = -abs(speed - self.desired_speed)
      
      # Reward for collision
      r_collision = 0
      if self.collision_detector.get_latest_collision_intensity() is not None:
        r_collision = -1

      # Reward for steering:
      r_steer = -self.ego.get_control().steer**2

      # Reward for out of lane
      ego_x, ego_y = get_pos(self.ego)
      dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
      r_out = 0
      if abs(dis) > self.out_lane_thres:
        r_out = -1

      # Longitudinal speed
      lspeed = np.array([v.x, v.y])
      lspeed_lon = np.dot(lspeed, w)

      # Cost for too fast
      r_fast = 0
      if lspeed_lon > self.desired_speed:
        r_fast = -1

      # Cost for lateral acceleration
      r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

      # Total reward combination
      total_reward = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

      # Log reward components to TensorBoard
      reward_components = {
        "speed_reward": r_speed,
        "collision_reward": r_collision,
        "steering_reward": r_steer,
        "out_of_lane_reward": r_out,
        "too_fast_reward": r_fast,
        "lateral_acceleration_reward": r_lat,
        "total_reward": total_reward
      }

      return total_reward, reward_components

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if self.collision_detector.get_latest_collision_intensity():
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True

    # If out of lane
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      return True

    return False

  def _get_info(self):
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # state information
    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front
    }
    return info
