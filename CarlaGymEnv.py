import glob
import os
import sys
import random
import time
import numpy as np
import math
import datetime
import collections
import re
import weakref

import copy
from PIL import Image

from gym import error, spaces

from collections import deque

from threading import Thread

import cv2
import pygame

import carla

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

from carla import ColorConverter as cc
from agents.navigation.global_route_planner import GlobalRoutePlanner       
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO




WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,

        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.CloudySunset,

        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset,

        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.MidRainSunset,

        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.WetCloudySunset,

        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.HardRainSunset,

        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
]




class CarEnv:

    def __init__(self, rank, sim_fps=10,
                    sensor_img_save=False,
                    render=None,
                    withlock=None,
                    show_preview=True,
                    hidden_obs=True,
                    im_width=300,
                    im_height=300,
                    secs_per_episode=30,
                    steer_amt=1.1,
                    auto_reset=True,
                    sparse=False,
                    dist_reward=False,
                    start_init=True,
                    benchmark="STDRandom",
                    step_type="skipping",
                    to_record=False,
                    discrete=True):

        print('Initializing Car Env')

        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(10.0)
        self.rank = rank
        self.render = render
        self.to_record = to_record
        self.sensor_img_save = sensor_img_save

        self.auto_reset = auto_reset

        self.show_cam = show_preview

        self.withlock = withlock

        self.dist_reward = dist_reward


        self.car_im_width = im_width
        self.car_im_height = im_height
        self.steer_amt = steer_amt
        self.secs_per_episode = secs_per_episode

        self.env_episode_numb = 0

        self.world = self.client.get_world()

        self.sim_fps = sim_fps

        self.record_video_state = 0
        self.to_record_frame = False
        
        self.change_weather_step = 100

        self.select_benchmarks(benchmark)

        self.sparse = sparse


        if(sparse):
            print('Carla in Sparse Mode')

        #settings = self.world.get_settings()
        #settings.fixed_delta_seconds = 0.5
        #self.world.apply_settings(settings)
        
        if(sim_fps != 0):
            self.delta_seconds = 1.0 / sim_fps
            self.frame = self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False, synchronous_mode=True, fixed_delta_seconds=self.delta_seconds))

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("leon")[0]

        #self.sensor_list = ["rgbcam1", "rgbcam2", "rgbcam3", "minimap"]

        #self.sensor_list = ["rgbcam1", "minimap"]
        #self.sensor_list = ["rgbcam1"]

        self.sensor_list = ["rgbcam1", "rgbback", "minimap"]

        self.amount_images = len(self.sensor_list)

        self.handle_obsspace()

        self.discrete = discrete

        if(discrete):
            self.action_space = spaces.Discrete(9)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2, 1), dtype=np.float32)

        self.vehicle = None
        self.player = self.vehicle

        if(step_type == "skipping"):
            self.step = self.step_frame_skipping
        elif(step_type == "single_no_obs"):
            self.step = self.step_only_image
        else:
            self.step = self.step_aug


        if(start_init):
            self.env_init()


    def handle_obsspace(self):
        final_height = self.car_im_height
        final_width = self.car_im_width*self.amount_images

        self.final_output = [0]*self.amount_images

        self.observation_space = spaces.Box(low=0, high=1, shape=(final_height, final_width, 3), dtype=np.float32)


        # Aditional Obs space includes things like speed, and indications from gps (has to be 5 of dim because of hot encoded directions)
        self.addtionals_obs_callbacks = [self.get_obs_speed, self.get_obs_indications]
        self.additional_obs_space = spaces.Box(low=0, high=1, shape=(len(self.addtionals_obs_callbacks), 6), dtype=np.float32)


    def add_sensors(self, sensors):
        self.static_index = 0
        for sensor in sensors:
            if(sensor == 'rgbcam1'):
                rgb_cam1 = self.blueprint_library.find('sensor.camera.rgb')
                rgb_cam1.set_attribute("image_size_x", f"{self.car_im_width}")
                rgb_cam1.set_attribute("image_size_y", f"{self.car_im_height}")
                rgb_cam1.set_attribute("fov", f"160")

                
                transform = carla.Transform(carla.Location(x=2.5, z=0.7))
                rgb_cam1 = self.world.spawn_actor(rgb_cam1, transform, attach_to=self.vehicle)
                what_index1 = int(self.static_index) + 1 - 1
                rgb_cam1.listen(lambda data: self.process_img(data, what_index1))
                self.static_index += 1
                self.actor_list.append((rgb_cam1))

            if(sensor == 'rgbcam2'):
                rgb_cam2 = self.blueprint_library.find('sensor.camera.rgb')
                rgb_cam2.set_attribute("image_size_x", f"{self.car_im_width}")
                rgb_cam2.set_attribute("image_size_y", f"{self.car_im_height}")
                rgb_cam2.set_attribute("fov", f"160")

                transform2 = carla.Transform(carla.Location(x=3.2, y=-1.4, z=0.7))
                rgb_cam2 = self.world.spawn_actor(rgb_cam2, transform2, attach_to=self.vehicle)
                what_index2 = int(self.static_index) + 1 - 1
                rgb_cam2.listen(lambda data: self.process_img(data, what_index2))
                self.static_index += 1
                self.actor_list.append((rgb_cam2))
            
            if(sensor == 'rgbcam3'):
                rgb_cam3 = self.blueprint_library.find('sensor.camera.rgb')
                rgb_cam3.set_attribute("image_size_x", f"{self.car_im_width}")
                rgb_cam3.set_attribute("image_size_y", f"{self.car_im_height}")
                rgb_cam3.set_attribute("fov", f"160")

                transform3 = carla.Transform(carla.Location(x=2.5, y=1.4, z=0.7))
                rgb_cam3 = self.world.spawn_actor(rgb_cam3, transform3, attach_to=self.vehicle)
                what_index3 = int(self.static_index) + 1 - 1
                rgb_cam3.listen(lambda data: self.process_img(data, what_index3))
                self.static_index += 1
                self.actor_list.append((rgb_cam3))
            
            
            if(sensor == 'rgbback'):
                rgbback = self.blueprint_library.find('sensor.camera.rgb')
                rgbback.set_attribute("image_size_x", f"{self.car_im_width}")
                rgbback.set_attribute("image_size_y", f"{self.car_im_height}")
                rgbback.set_attribute("fov", f"160")

                transform5 = carla.Transform(location=carla.Location(x=-2.5, y=0.0, z=2), rotation=carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0))
                rgbback = self.world.spawn_actor(rgbback, transform5, attach_to=self.vehicle)
                what_index3 = int(self.static_index) + 1 - 1
                rgbback.listen(lambda data: self.process_img(data, what_index3))
                self.static_index += 1
                self.actor_list.append((rgbback))
        

            if(sensor == 'minimap'):
                minimap = self.blueprint_library.find('sensor.camera.rgb')
                minimap.set_attribute("image_size_x", f"{self.car_im_width}")
                minimap.set_attribute("image_size_y", f"{self.car_im_height}")
                minimap.set_attribute("fov", f"50")

                transform4 = carla.Transform(location=carla.Location(x=0.0, y=0.0, z=100.0), rotation=carla.Rotation(pitch=270.0, yaw=0.0, roll=0.0))
                minimap = self.world.spawn_actor(minimap, transform4, attach_to=self.vehicle)
                what_index4 = int(self.static_index) + 1 - 1
                minimap.listen(lambda data: self.process_img(data, what_index4))
                self.static_index += 1
                self.actor_list.append((minimap))


        lidar = self.blueprint_library.find('sensor.lidar.ray_cast')
        #lidar.set_attribute("image_size_x", f"{self.car_im_width}")
        #lidar.set_attribute("image_size_y", f"{self.car_im_height}")
        #lidar.set_attribute("fov", f"50")

        lidar.set_attribute('channels',str(32))
        lidar.set_attribute('points_per_second',str(900000))
        lidar.set_attribute('rotation_frequency',str(100))
        lidar.set_attribute('range',str(50))
        #lidar.set_attribute('UpperFOVLimit',str(10))
        #lidar.set_attribute('LowerFOVLimit',str(-30))
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        #lidar = self.world.spawn_actor(lidar, lidar_transform, attach_to=self.vehicle)
        #what_index4 = int(self.static_index) + 1 - 1
        #lidar.listen(lambda data: self.process_lidar(data))
        #self.actor_list.append((lidar))



    def process_lidar(self, data):
        if(self.global_step_numb % 100 == 0):
            data.save_to_disk('./data/lidar/%.6d.ply' % data.frame)


    def env_init(self):

        if(self.vehicle is not None):
            self.world.destroy()
            for act in self.actor_list:
                act.destroy()

        self.collision_hist = []
        self.actor_list = []

        self.transform, self.destination = random.sample(self.world.get_map().get_spawn_points(), 2)

        self.init_pos, self.destination = self.transform.location, self.destination.location

        self.destination = carla.Location(x=0, y=0, z=0) # Make all cars go to (0,0,0)

        self.last_ddest = math.sqrt((self.init_pos.x-self.destination.x)**2 + (self.init_pos.y-self.destination.y)**2 + (self.init_pos.z-self.destination.z)**2)

        #print('Distance to Destination (0,0,0): {}'.format(self.last_ddest))

        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.vehicle.set_simulate_physics(True)

        self.actor_list.append(self.vehicle)

        self.add_sensors(self.sensor_list) 

        self.image_index = 0

        self.global_step_numb = 0

        if(self.rank == 0):
            self.world.tick()
  
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        self.step_numb = 0
    
        if(self.rank == 0):
            self.world.tick()

        time.sleep(0.3)

        #self.world.restart(self.vehicle)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))


        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self.lane_data(event))

        self.lane_inv_hist = []

        #while self.final_output[0] is 0:
        #    time.sleep(0.005)

        time.sleep(0.005)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        if(self.rank == 0):
            self.world.tick()

        self.last_col = None

        return self.make_final_output()


    def reset(self):

        print('Resetting')

        self.collision_hist = []

        #self.transform, self.destination = random.sample(self.world.get_map().get_spawn_points(), 2)

        #self.init_pos = self.transform.location
        #self.init_rot = self.transform.rotation

        #self.destination = carla.Location(x=0, y=0, z=0) # Make all cars go to (0,0,0)

        self.init_pos, self.init_rot, self.destination = self.init_dest_waypoint_get(self)

        self.last_ddest = math.sqrt((self.init_pos.x-self.destination.x)**2 + (self.init_pos.y-self.destination.y)**2 + (self.init_pos.z-self.destination.z)**2)

        #print('Distance to Destination (0,0,0): {}'.format(self.last_ddest))

        self.vehicle.set_simulate_physics(False)

        self.vehicle.set_transform(carla.Transform(self.init_pos, self.init_rot))
        self.vehicle.set_angular_velocity(carla.Vector3D(0,0,0))

        self.image_index = 0

        if(self.rank == 0):
            self.world.tick()
  
        self.step_numb = 0

        self.lane_inv_hist = []

        self.episode_start = time.time()

        if(self.rank == 0):
            self.world.tick()

        self.vehicle.set_simulate_physics(True)

        self.last_col = None

        self.astar_wapoints_tolocs(self.init_pos, self.destination)

        self.visited_locs = []

        self.env_episode_numb += 1

        return self.make_final_output()


    def change_weather(self, step):
        index = (step // 100) % len(WEATHERS)
        self.world.set_weather(WEATHERS[index])


    def record_video(self, time_secs=38):
        # 0 -> Start to record, 1 -> recording, see if time reached end and save
        if(self.to_record == False):
            return

        print('Record Video called {}'.format(self.record_video_state))

        if(self.record_video_state == 0):
            self.time_to_record = time_secs
            self.fourcc = cv2.VideoWriter_fourcc(*'MP42')
            self.start_video_secs = round(time.time())
            name = 'CarlaEnvVid_{}.avi'.format(self.start_video_secs)
            self.video = cv2.VideoWriter(name, self.fourcc, float(20), (900, 300))
            self.record_video_state = 1
            self.to_record_frame = True
            print('Going to record a video')

        if(self.record_video_state == 1):
            if(time.time() > (self.start_video_secs + self.time_to_record)):
                self.record_video_state = 0
                self.video.release()
                self.to_record_frame = False



    def select_benchmarks(self, benchmark):


        def stdrand_waypoints_pos(env):
            transform = random.sample(env.world.get_map().get_spawn_points(), 1)[0]
            init_pos = transform.location
            init_rot = transform.rotation
            destination = carla.Location(x=0, y=0, z=0) # Make all cars go to (0,0,0)
            return  init_pos, init_rot, destination


        def stdfixed_waypoints_pos(env):
            transform = env.world.get_map().get_spawn_points()[1]
            init_pos = transform.location
            init_rot = transform.rotation
            destination = carla.Location(x=0, y=0, z=0) # Make all cars go to (0,0,0)
            return  init_pos, init_rot, destination


        def std_benchmark_step_hook(env):
            #print('On STD benchmark hook')
            #print(env.env_episode_numb)
            return 0, False # Return reward, and Done


        if(benchmark == 'STDRandom'):
            self.benchmakr_step_hook = std_benchmark_step_hook
            self.init_dest_waypoint_get = stdrand_waypoints_pos
        if(benchmark == 'STDFixed'):
            self.benchmakr_step_hook = std_benchmark_step_hook
            self.init_dest_waypoint_get = stdfixed_waypoints_pos


    def collision_data(self, event):
        self.collision_hist.append(event)


    def lane_data(self, event):
        self.lane_inv_hist.append(event)


    def tickworld(self):
        if(not self.withlock):
            self.world.tick()
        else:
            self.lock.acquire()
            self.world.tick()
            self.lock.release()


    def process_img(self, image, numb):
       
        i = np.array(image.raw_data)

        i2 = i.reshape((self.car_im_height, self.car_im_width, 4))

        i3 = i2[:, :, :3]
        
        self.final_output[numb] = i3


    def make_final_output(self):
        
        final_img = self.final_output[0]

        for image in range(1, len(self.final_output)):
            final_img = np.hstack((final_img, self.final_output[image]))


        if self.sensor_img_save:
            im = Image.fromarray(final_img)
            im.save('images/' + str(time.time()).replace('.', '') + '.png')
            
        if self.show_cam:
            pass
            #
            #cv2.imshow("CarlaGymEnv Cam", final_img)
            #cv2.waitKey(1)


        #final_img = final_img.transpose((-1, 0, 1)).astype(np.float32)/255 # Why this
        
        final_img = final_img.astype(np.float32)/255

        if(self.to_record_frame):
            frame_np = final_img*255
            frame_np = frame_np.astype(np.uint8)
            self.video.write(frame_np)
            self.record_video()

        #return [final_img, np.asarray(self.last_ddest)]
        return [final_img, self.additional_obs()]


    def additional_obs(self):
        additional_obs = np.random.rand(len(self.addtionals_obs_callbacks), 6).astype('float32')

        for i in range(len(self.addtionals_obs_callbacks)):
            to_call = self.addtionals_obs_callbacks[i]
            additional_obs[0] = to_call()

        #print(additional_obs)

        return additional_obs


    def get_obs_speed(self):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        return kmh


    def get_obs_indications(self):
        hot_encoded = np.asarray([0, 0, 0, 0, 0, 0], dtype='float32')
        if hasattr(self, "indication"):
            # Road indication comes from 1-6 also need to make sure that if not an int call .value to it
            if(not isinstance(self.indication, int)):
                self.indication = self.indication.value
            hot_encoded[self.indication - 1] = 1.0
        else:
            hot_encoded[0] = 1.0
        
        return hot_encoded


    def render(self):
        print('Rendering image to screen')
        #cv2.imshow("", self.front_camera)


    def is_near_junc(self, vehicle, dis=10, debug=False):
        veh_loc_way = world.map.get_waypoint(vehicle)
        
        junc = None

        to_check_waypoints = veh_loc_way.next(dis)
        for way in to_check_waypoints:

            if(way.is_junction):
                junc = way
                break

        if(junc is None and debug):
            print('Vehicle: {} is near {} dis of junction {}'.format(str(vehicle), str(dis), str(junc.id)))

        return is_junc


    def calc_distance(self, loc1, loc2):
        return math.sqrt(
                (loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 + (loc1.z - loc2.z) ** 2)


    def calc_junction_risk(self):

        vehicles = self.world.get_actors().filter('vehicle.*')

        my_junc = is_near_junc(self.vehicle)

        if(my_junc is None or len(vehicles) <= 1):
            print('I am not in a junction or 1 vehicle')
            return 0

        all_rewards = 0

        vehicles_near_juncs = []

        for ve in vehicles:
            if(ve != self.vehicle and is_near_junc(ve) == my_junc):
                vehicles_near_juncs.append(ve)


        if(len(vehicles_near_juncs)):
            print('Not a single other vehicle is near my junctions')
            return 0

        vego = self.vehicle.get_velocity()

        veh_ego_way = self.world.map.get_waypoint(self.vehicle)

        for other_vehicle in vehicles_near_juncs:

            ov_loc = other_vehicle.get_location()
            ov_waypoint = self.world.map.get_waypoint(ov_loc)

            collision_points = list(set(ov_waypoint.next_until_lane_end(20)) & set(self.vehicle.next_until_lane_end(20))) 

            if(len(collision_points) == 0):
                print('No collsion with this vehicle')
                continue

            collision_point = collision_points[0].transform.location

            v_oi = other_vehicle.get_velocity()

            doi = calc_distance(ov_loc, collision_point)



    def astar_wapoints_tolocs(self, start_loc, end_loc, draw_waypoints=True):

        #end_waypoint = env.world.map.get_waypoint(carla.Location(0, 0, 0))
        #start_waypoint = env.world.map.get_waypoint(car.get_location())   

        map_ = self.world.get_map()
        dao = GlobalRoutePlannerDAO(map_, 5.0)
        grp = GlobalRoutePlanner(dao)                 
        grp.setup()
        #self.route_wayp_opts = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)

        self.route_wayp_opts = grp.trace_route(start_loc, end_loc)

        if(len(self.route_wayp_opts) > 1):
            self.route_wayp_opts.pop(0)

        #import pudb; pudb.set_trace()

        self.route_locs_opts = []

        for w_opt_tuple in  self.route_wayp_opts:

            loc = w_opt_tuple[0].transform.location
            opt = w_opt_tuple[1].value

            self.route_locs_opts.append((loc, opt))

            if(draw_waypoints):
                t = w_opt_tuple[0].transform
                begin = t.location + carla.Location(z=0.5)
                angle = math.radians(t.rotation.yaw)
                end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))

                color = carla.Color(255,0,0)
                if(w_opt_tuple == self.route_wayp_opts[0]):
                    color = carla.Color(0,255,0)

                self.world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=15, color=color)

        self.indication = self.route_wayp_opts[0][1]



        #print("Calculated A* route got {:3d} waypoints".format(len(self.route_locs_opts)))


    def cal_vel_reward(self, cruise_speed=40):
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        reward  = np.clip(kmh, 0, 40)*1.5/cruise_speed

        if(kmh < 8):
            reward = -0.5
 
        return reward, False


    def cal_time_reward(self, max_steps=1000):

        #done = False

        #if(self.step_numb >= max_steps or self.episode_start + self.secs_per_episode < time.time()):
        #    done = True

        #return self.step_numb/max_steps, done

        return -0.1, False


    def calc_waypoints_reward(self, min_dis=1.2, debug=False):

        ## 1-> Left, 2 -> Right , 3 -> Straight, 4 -> LaneFollow, 5 -> ChangeLaneLeft
        reward = 0

        car_loc = self.vehicle.get_location()

        #self.visited_locs = []

        #print(self.route_locs_opts)
            
        nearest_proposed = self.route_locs_opts[0]
        indication = nearest_proposed[1]
        if(len(self.route_locs_opts) > 1):
            nearest_proposed2 = self.route_locs_opts[1]
        else:
            return 0

        if(debug):
            print('I am here ({:3.2f}, {:3.2f}, {:3.2f}) propsed here {:3.2f}, {:3.2f}, {:3.2f} and here {:3.2f}, {:3.2f}, {:3.2f}'.format(car_loc.x, car_loc.y, car_loc.z, nearest_proposed[0].x, nearest_proposed[0].y, nearest_proposed[0].z, nearest_proposed2[0].x, nearest_proposed2[0].y, nearest_proposed2[0].z))

        if(self.locs_dist(nearest_proposed[0], car_loc) < min_dis):
            self.route_locs_opts.remove(nearest_proposed)
            self.visited_locs.append(nearest_proposed[0])
            indication = nearest_proposed2[1]
            if(debug):
                print('Got to a proposed waypoint')
            reward += 1
        elif(self.locs_dist(nearest_proposed2[0], car_loc) < min_dis):
            self.route_locs_opts.remove(nearest_proposed2)
            self.route_locs_opts.remove(nearest_proposed)
            self.visited_locs.append(nearest_proposed2[0])
            indication = self.route_locs_opts[3][1]
            if(debug):
                print('Got to a proposed waypoint')
            reward += 1


        if(debug):
            print('I am indication to go: {}'.format(indication))

        self.indication = indication

        return reward

    
    def locs_dist(self, loc1, loc2):
        return math.sqrt((loc1.x-loc2.x)**2 + (loc1.y-loc2.y)**2 + (loc1.z-loc2.z)**2)

    def cal_dis_reward(self, max_dis=100):

        new_d_dest = math.sqrt((self.vehicle.get_location().x-self.destination.x)**2 + (self.vehicle.get_location().y-self.destination.y)**2 + (self.vehicle.get_location().z-self.destination.z)**2)
        
        if(new_d_dest <= 30):
            print('Got to the objective')
            return 1, True

        if(self.step_numb <= 4 ):
            self.initial_dist = new_d_dest
            self.last_ddest = new_d_dest
            return 0, False


        dist_to_dest_trav = self.last_ddest - new_d_dest

        if(dist_to_dest_trav < -0.1): # allow for small backwards movements
            dist_to_dest_trav *= np.clip(self.step_numb, 3, 10)

        reward = dist_to_dest_trav/(self.initial_dist/4)


        self.last_ddest = new_d_dest

        #print('Distance Reward: {}'.format(reward))

        return reward, False


    def cal_collision_reward(self, debug=False):

        reward, done = 0, False

        if len(self.collision_hist) != 0:
            impulse = self.collision_hist[-1].normal_impulse
            impulse = math.sqrt(impulse.x**2 + impulse.y**2 +  impulse.z**2)

            if(impulse > 300 or len(self.collision_hist) > 5):
                if(debug):
                    print('Done because of collision: amount: {}'.format(impulse))
                done = True
                reward = -1
            else:
                #print('Collided but it wasnt strong enough to end it col_hist: {}.'.format(len(self.collision_hist)))

                #Stops lingering collisions giving bad rewards
                col_timestamp = int(self.collision_hist[-1].timestamp*1000)
                #print('Col ts: {}, last ts: {}'.format(col_timestamp, self.last_col))
                if(col_timestamp == self.last_col):
                    #print('POPPED a lingering collision')
                    self.collision_hist.pop()
                    done = False
                else:

                    done = False
                    reward += -0.5 #avoid it rather than finish the episode and be done with it

                self.last_col = col_timestamp

        return reward, done

    def cal_lane_reward(self, debug=False):


        """
        waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
        print("Current lane type: " + str(waypoint.lane_type))
        # Check current lane change allowed
        print("Current Lane change:  " + str(waypoint.lane_change))
        # Left and Right lane markings
        print("L lane marking type: " + str(waypoint.left_lane_marking.type))
        print("L lane marking change: " + str(waypoint.left_lane_marking.lane_change))
        print("R lane marking type: " + str(waypoint.right_lane_marking.type))
        print("R lane marking change: " + str(waypoint.right_lane_marking.lane_change))
        """

        legal_crosses = [carla.libcarla.LaneMarkingType.BrokenBroken, carla.libcarla.LaneMarkingType.Broken, carla.libcarla.LaneMarkingType.NONE,
                         carla.libcarla.LaneMarkingType.Other, carla.libcarla.LaneMarkingType.BottsDots]

        if len(self.lane_inv_hist) != 0:
            for crossed_mark in self.lane_inv_hist[-1].crossed_lane_markings:
                if(debug):
                    print(crossed_mark.type, end=", ")
                if(crossed_mark.type not in legal_crosses):
                    if(debug):
                        print("Passed an illegal lane, type: {}, ending episode".format(crossed_mark.type))
                    return -2, True
                else:
                    if(debug):
                        print("Passed a legal lane, type: {}".format(crossed_mark.type))
            
            self.lane_inv_hist.pop()

        return 0, False


    def calc_lights_reward(self, debug=True):

        vehicle_actor = self.vehicle

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        reward, done = 0, False

        if vehicle_actor.is_at_traffic_light():
            traffic_light = vehicle_actor.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                if(debug):
                    print("At a redlight")

                if(kmh < 5):
                    reward += 0.75
                elif(kmh > 20):
                    reward, done = -1, True
            else:
                if(debug):
                    print("At a yellow or green light")

                if(kmh > 8):
                    reward += 0.75

        return reward, done

    def calculate_reward(self, vel_w=0.1, junc_w=0.3, time_w=-0.07, distance_w=0.8, collision_w=2, lane_w=2, lights_w=2, lights_thres=1000, debug=False):

        # All rewards should try be -1 to 1.
        
        done, reward, info = False, 0, None

        dist_l_reward = []

        vel_reward, v_done = self.cal_vel_reward(cruise_speed=40)

        reward += vel_w * vel_reward
        done |= v_done
        dist_l_reward.append(vel_reward)

        t_reward, t_done = self.cal_time_reward(max_steps=1000)

        reward += time_w * t_reward
        done |= t_done
        dist_l_reward.append(t_reward)

        d_reward, d_done = self.cal_dis_reward(max_dis=100)

        reward += distance_w * d_reward
        done |= d_done
        dist_l_reward.append(d_reward)

        col_reward, col_done = self.cal_collision_reward()

        reward += col_reward * collision_w
        done |= col_done
        dist_l_reward.append(col_reward)

        lane_reward, lane_done = self.cal_lane_reward()

        reward += lane_reward * lane_w
        done |= lane_done
        dist_l_reward.append(lane_reward)

        if(self.env_episode_numb > lights_thres):
            lights_reward, lights_done = self.calc_lights_reward()

            reward += lights_reward * lights_w
            done |= lights_done


        # Benchmark/Scenario Handling

        bench_rew, bench_done = self.benchmakr_step_hook(self)

        if(bench_done):
            print('Benchmark Done, Final reward {}'.format(bench_rew))
            return bench_rew, True, {'scen_sucess':1, 'scen_metric':bench_rew}


        reward += self.calc_waypoints_reward()

        if(debug):
            print("Vel_r : {:2.2f}, Time_r : {:2.2f}, Dis_r : {:2.2f}, Col_r : {:2.2f}, Lan_r : {:2.2f}".format(vel_reward, t_reward, d_reward, col_reward, lane_reward))


        if(self.sparse):
            if(d_done):
                return 1, True, {'scen_sucess':1, 'scen_metric':bench_rew}
            elif(done):
                return -1, True, {'scen_sucess':-1, 'scen_metric':-1}
            else:
                return 0, False, info

        if(d_done):
            info = {'scen_sucess':1, 'scen_metric':-1}

        if(done == True):
            if(info == None):
                info = {'scen_sucess':-1, 'scen_metric':-1}


        if(self.dist_reward):
            return tuple(dist_l_reward), done, info
        else:
            return reward, done, info


    def _step(self, action, continuos=False):

        self.step_numb += 1
        self.global_step_numb += 1

        if(self.global_step_numb % self.change_weather_step == 0):
            self.change_weather(self.global_step_numb)

        self.tickworld()

        if(continuos == False):

            if(isinstance(action, int)):
                action = int(action/3), action%3

            thrt_action, steer_action = action
            #Discrete(3) -> 0, 1, 2 -> transform to -1, 0, 1
            thrt_action -= 1
            steer_action -= 1
            self.vehicle.apply_control(carla.VehicleControl(throttle=thrt_action, steer=steer_action*self.steer_amt))
        else:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=action[0][0], steer=action[1][0]))
            self.vehicle.apply_control(carla.VehicleControl(throttle=action[0].item(), steer=action[1].item()))
        
        reward, done, info = self.calculate_reward()

        osb = self.make_final_output()

        if(self.auto_reset and done):
            self.reset()

        return osb, reward, done, info


    def step_frame_skipping(self, action, numb_frames=4, continuos=False):
        final_reward, final_done = 0, False

        frames = []

        obs_i, rw_i, done_i, info = self._step(action, continuos)

        frames.append(obs_i)
        final_reward += rw_i
        final_done |= done_i

        for f_i in range(numb_frames - 1):
            obs_i, rw_i, done_i, info = self._step((action), False)

            frames.append(obs_i)
            final_reward += rw_i
            final_done |= done_i


        return frames, final_reward, final_done, info


    def step_only_image(self, action, continuos=False):
        obs, reward, done, info = self._step(action, not self.discrete)
        return obs[0], reward, done, info


    def step_aug(self, action, continuos=False):
        obs, reward, done, info = self._step(action, not self.discrete)
        return obs, reward, done, info

    def seed(self, numb):
        np.random.seed(numb)
        random.seed(numb)

    #Testing
    def _go(self, what=(2,1), steps=20):
        for i in range(steps):
            res = self.step(what)
            print("{:1.3f}".format(res[1]), end=" ")
            print(res[2])

    def numb_actions(self):
        return 3


    def lower(self):
        return 'car_env'

    def __del__(self):
        print('Carla Descontructor Called')
        #self.(synchronous_mode=False)

        if(self.vehicle is not None):
            self.world.destroy()
            for act in self.actor_list:
                act.destroy()


    @classmethod
    def make(cls, args=None):
        return cls()




class CarEnvScenario(CarEnv):

    def __init__(self, rank):

        #super(CarEnvScenario, self).__init__(rank, sparse=True, start_init=False)
        super(CarEnvScenario, self).__init__(rank, render=True, step_type="other", benchmark="STDRandom", auto_reset=False, discrete=False, sparse=True, start_init=False, sim_fps=0)

        #import pudb; pudb.set_trace()
        
        while self.vehicle is None:
            print("Scenario not yet ready")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    self.vehicle = vehicle

        self._server_clock = pygame.time.Clock()

        self.world.on_tick(self.on_tick_stub)
        self.world.wait_for_tick(10.0)

        self.collision_hist = []
        self.actor_list = []

        self.transform, self.destination = random.sample(self.world.get_map().get_spawn_points(), 2)

        self.init_pos, self.destination = self.transform.location, self.destination.location

        self.destination = carla.Location(x=0, y=0, z=0) # Make all cars go to (0,0,0)

        self.last_ddest = math.sqrt((self.init_pos.x-self.destination.x)**2 + (self.init_pos.y-self.destination.y)**2 + (self.init_pos.z-self.destination.z)**2)

        #print('Distance to Destination (0,0,0): {}'.format(self.last_ddest))

        #self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        #self.vehicle.set_simulate_physics(True)

        #self.actor_list.append(self.vehicle)

        self.add_sensors(self.sensor_list) 

        self.image_index = 0

        if(self.rank == 0):
            self.world.tick()
  
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        self.step_numb = 0
    
        if(self.rank == 0):
            self.world.tick()

        time.sleep(0.3)

        #self.world.restart(self.vehicle)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))


        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, carla.Transform(carla.Location(x=2.5, z=0.7)), attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self.lane_data(event))

        self.lane_inv_hist = []

        #while self.final_output[0] is 0:
        #    time.sleep(0.005)

        time.sleep(0.005)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        if(self.rank == 0):
            self.world.tick()

        self.last_col = None

        #return self.make_final_output()

    def calculate_reward(self):
        return 0, False

    def reset(self):
        return self.make_final_output()

    def on_tick_stub(self, timestamp):

        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds



