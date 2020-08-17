import glob
import time
from collections import namedtuple, deque
from random import randrange, uniform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

import random

import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal

from torch.distributions import Categorical, Normal
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchviz import make_dot

import cv2
from PIL import Image, ImageDraw

import wandb

import threading
import queue


import os
os.environ["WANDB_MODE"] = "dryrun"


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import re
import time
import weakref


import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_0
from pygame.locals import K_9
from pygame.locals import K_BACKQUOTE
from pygame.locals import K_BACKSPACE
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_c
from pygame.locals import K_d
from pygame.locals import K_h
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_w



class World(object):
    def __init__(self, carla_world, hud, vehicle):
        self.world = carla_world
        self.mapname = carla_world.get_map().name
        self.hud = hud
        self.world.on_tick(hud.on_world_tick)
        #self.world.wait_for_tick(10.0)
        self.vehicle = vehicle

        self.vehicle_name = self.vehicle.type_id
        #self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        #self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager.set_sensor(0, notify=False)
        self.controller = None
        #self._weather_presets = find_weather_presets()
        #self._weather_index = 0


    """
    def restart(self):
        cam_index = self.camera_manager._index
        cam_pos_index = self.camera_manager._transform_index
        start_pose = self.vehicle.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        blueprint = self._get_random_blueprint()
        self.destroy()
        self.vehicle = self.world.spawn_actor(blueprint, start_pose)
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.vehicle)
        self.hud.notification(actor_type)

    """

    def tick(self, clock):
        #if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
        #    print("Scenario ended -- Terminating")
        #    return False

        self.hud.tick(self, self.mapname, clock)
        return True

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()




class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, mapname, clock):
        if not self._show_info:
            return
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        #colhist = world.collision_sensor.get_collision_history()
        #collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        #max_col = max(1.0, max(collision))
        #collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            #'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            'Map:     % 20s' % mapname,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake)]
            #'Collision:',
            #collision,
            #'',
            #'Number of vehicles: % 8d' % len(vehicles)
        
        """
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        """
        self._notifications.tick(world, clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:

            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        #self.help.render(display)





class KeyboardControl(object):
    def __init__(self, world, queue, start_in_autopilot=False):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        #world.vehicle.set_autopilot(self._autopilot_enabled)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.queue = queue

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                #elif event.key == K_BACKSPACE:
                    #world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                #elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                #    world.next_weather(reverse=True)
                #elif event.key == K_c:
                #    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.reverse = not self._control.reverse
                #elif event.key == K_p:
                #    self._autopilot_enabled = not self._autopilot_enabled
                #    world.vehicle.set_autopilot(self._autopilot_enabled)
                #    world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            #print('Parsing Events')
            self._parse_keys(pygame.key.get_pressed(), clock.get_time())
            #world.vehicle.apply_control(self._control)

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0

        self._control.throttle = -1.0 if keys[K_DOWN] or keys[K_s] else self._control.throttle

        #print('milliseconds: {}'.format(milliseconds))
        steer_increment = 5e-4 * 30
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        #print('Steer Cache: {}'.format(self._steer_cache))
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

        #print('Control Steer: {}, Throttle: {}'.format(self._steer_cache, self._control.throttle))
        self.queue.put(np.asarray([self._control.throttle, self._control.steer]))


    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)








class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)



# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self._hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)















def read_kbd_input(inputQueue):
    print('Ready for keyboard input:')
    while (True):
        # Receive keyboard input from user.
        input_str = input()

        # Enqueue this input string.
        # Note: Lock not required here since we are only calling a single Queue method, not a sequence of them 
        # which would otherwise need to be treated as one atomic operation.
        inputQueue.put(input_str)
        time.sleep(0.05)


class HumanAgent():

    def __init__(self, env, obs_state, num_actions, hyperps, device):
        self.hyperps = hyperps
        self.env = env
        self.device = device

        self.num_actions = num_actions

        self.inputQueue = queue.Queue()



        pygame.init()
        pygame.font.init()


        #client = carla.Client(args.host, args.port)
        #client.set_timeout(2.0)

        self.display = pygame.display.set_mode(
                (640, 480),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.hud = HUD(640, 480)

        #import pudb; pudb.set_trace()

        self.world = World(env.client.get_world(), self.hud, env.vehicle)

        self.controller = KeyboardControl(self.world, self.inputQueue)
        self.clock = pygame.time.Clock()
        
        #while True:




    def sample(self, obs):


        self.clock.tick_busy_loop()
        #if self.controller.parse_events(self.world, self.clock):
        if self.controller.parse_events(self.world, self.clock):
            return
        #if not self.world.tick(self.clock):
        #    return
        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()        


        #pil_obs = transforms.ToPILImage()(obs[0][0])


        #time.sleep(1.25)


        #_draw = ImageDraw.Draw(pil_obs)

        #_draw.text((5, 10), 'FPS: %.3f, steps: %.3f' % (steps / (time.time() - wall_start), steps))

        #_draw.text((5, 30), 'Steer: %.3f' % all_pol_stats[-1][0])

        #_draw.text((5, 50), 'Throttle: %.3f' % all_pol_stats[-1][1])



        #import pudb; pudb.set_trace()


        #_combined = Image.fromarray(np.hstack((plot_img, np.asarray(pil_obs))))
        
        action = self.inputQueue.get()
        #print("input_str = {}".format(action))

        #cv2.imshow('Sensors', np.asarray(pil_obs))
        #cv2.waitKey(1)

        return action






def run_human_gathering(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=True, metrified=True, save_dir='./human_samples/', to_save=True):


    env.reset()
    human_agent = HumanAgent(env, obs_state, num_actions, hyperps, device)

    save_dir = './human_samples_lidar/'

    wall_start = time.time()

    total_steps = 1195
    updates = 0

    log_reward = []
    
    all_actions = []
    all_pol_stats = []
    all_stds = []
    all_means = []

    all_rewards = []
    all_scenario_wins_rewards = []
    all_final_rewards = []
    all_q_vals = []

    to_plot = []

    #[array([0.03867785]), array([-1.7760651]), array([0.06253806]), array([-5.08939411]), array([-0.1565633])]
    
    w_vel, w_t, w_dis, w_col, w_lan, w_waypoint = 0.5, 1, 1, 1, 1, 5

    rewards_weights = [w_vel, w_t, w_dis, w_col, w_lan, w_waypoint]
    change_rate = 0.1

    system_of_eqs = []

    for epi in range(hyperps['max_epochs']):
        obs = env.reset()
        
        old_obs = obs
        #old_obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))
        #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).float()
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}'.format(epi, hyperps['max_epochs']))
 

        for step_numb in range(hyperps['max_steps']):

            action = human_agent.sample(old_obs)

            #action = np.asarray([1.0, 1.0])

            obs, reward, done, info = env.step(action)

            all_rewards.append(reward)


            #print('Reward: Vel: {:.5f}, time: {:.5f}, dis: {:.5f}, col: {:.5f}, lan: {:.5f}'.format(w_vel*reward[0], w_t*reward[1], w_dis*reward[2],  w_col*reward[3], w_lan*reward[4]))
            
            #reward = (w_vel*reward[0] + w_t*reward[1] + w_dis*reward[2] + w_col*reward[3] + w_lan*reward[4])/5
        
            #print('Final Sum Reward: {:.5f}'.format(reward))



            print('Reward: Vel: {:.5f}, time: {:.5f}, dis: {:.5f}, col: {:.5f}, lan: {:.5f}, waypoint: {:.5f}'.format(w_vel*reward[0], w_t*reward[1], w_dis*reward[2],  w_col*reward[3], w_lan*reward[4], w_waypoint*reward[5]))
            
            #wandb.log({'reward_vel':w_vel*reward[0], 'reward_time':w_t*reward[1], 'reward_dist':w_dis*reward[2], 'reward_col':w_col*reward[3], 'reward_lane':w_lan*reward[4], 'reward_waypoint':w_waypoint*reward[5]})

            #reward = (w_vel*reward[0] + w_t*reward[1] + w_dis*reward[2] + w_col*reward[3] + w_lan*reward[4] + w_waypoint*reward[5])/6



            all_final_rewards.append(reward)

            if(info != None):
                print(info)
           

            #all_pol_stats.append([action.cpu().detach().numpy()[0][0], action.cpu().detach().numpy()[0][1], (log_prob[0]).item(), torch.clamp(torch.exp(log_prob[0]), 0, 1.0).item()])
            #all_means.append([mean.cpu().detach().numpy()[0][0], mean.cpu().detach().numpy()[0][1]])
            #all_stds.append([std.cpu().detach().numpy()[0][0], std.cpu().detach().numpy()[0][1]])


            


            if(to_save):
                with open(save_dir+'trans_{}.npz'.format(total_steps), 'wb') as f:
                    #np.savez(f, )
                    #np.savez(f, action)
                    #np.savez(f, np.asarray(reward))
                    #np.savez(f, obs[0])
                    #np.savez(f, obs[1])
                    #np.savez(f, np.asarray(done))
                    if(info != None): #{'scen_sucess':1, 'scen_metric':-1}
                        #np.savez(f, np.asarray([info['scen_sucess'], info['scen_metric']]))
                        print(info)
                        np.savez(f, (old_obs[0]*255).astype(np.uint8), old_obs[1], action, np.asarray(reward), (obs[0]*255).astype(np.uint8), obs[1], np.asarray(done), np.asarray([info['scen_sucess'], info['scen_metric']]))
                    else:
                        #np.savez(f, np.asarray([0, 0]))
                        
                        pass
                        np.savez(f, (old_obs[0]*255).astype(np.uint8), old_obs[1], action, np.asarray(reward), (obs[0]*255).astype(np.uint8), obs[1], np.asarray(done), np.asarray([0, 0]))


            #obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))

            #memory.push((old_obs[0].to("cpu"), old_obs[1].to("cpu")), action, reward, (obs[0].to("cpu"), obs[1].to("cpu")), done)

            #if(total_steps % 800 > 700):
            #    get_saliency(obs, sac_agent.actor, action.cpu().detach().numpy()[0], std, device)
            #    metrify(obs, total_steps, wall_start, np.asarray(all_actions), np.asarray(all_pol_stats), np.asarray(all_stds), np.asarray(all_means), np.asarray(all_rewards), np.asarray(all_scenario_wins_rewards), np.asarray(all_final_rewards), np.asarray(all_q_vals), to_plot)
                
            old_obs = obs

            total_steps += 1
            log_reward.append(reward)

            if done:
                #print('In SAC Done: {}'.format(len(memory)))

                if(info['scen_sucess'] != None and info['scen_sucess'] == 1):
                    all_scenario_wins_rewards.append(1)
                    
                    #to_plot.append([total_steps, info['scen_sucess']*0.999])

                elif (info['scen_sucess'] != None and info['scen_sucess'] == -1):
                    all_scenario_wins_rewards.append(-1)

                break

            done |= total_steps == hyperps['max_steps']


"""


"""