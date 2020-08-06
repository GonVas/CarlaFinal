import glob


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np



import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw

from carla_project.src.common import CONVERTER, COLOR
from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController






class Policy(nn.Module):

    def __init__(self, state_size, action_size, aditional_aug, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(Policy, self).__init__()

        self.state_size, channels_in = state_size
        self.action_size = action_size
        self.aditional_aug = aditional_aug

        #self.max_batch_size = max_batch_size

        self.conv1 = nn.Conv2d(channels_in, conv_channels, kernel_size, stride=1)

        self.size_now = self.conv_output_shape(self.state_size) 

        self.pool1 = nn.MaxPool2d(2, 2)

        self.size_now = (int(self.size_now[0]/2), int(self.size_now[1]/2))

        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size)

        self.size_now = self.conv_output_shape(self.size_now)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.size_now = int(self.size_now[0]/2) * int(self.size_now[1]/2) * conv_channels*2

        self.fc1 = nn.Linear(self.size_now + aditional_aug, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.final_layer = nn.Linear(size_3, action_size)

        self.critic = nn.Linear(size_3, 1)


    def forward(self, x):
        #import pudb; pudb.set_trace()

        x, aditional = x

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        
        x = x.reshape(-1, self.size_now)

        #import pudb; pudb.set_trace()
        x = torch.cat((x, aditional.reshape(-1, 12)), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        action = F.softmax(self.final_layer(x))

        value = F.relu(self.critic(x))

        return Categorical(action), value


    def conv_output_shape(self, h_w, kernel_size=3, stride=1, pad=0, dilation=1):
        """
        Utility function for computing output of convolutions
        takes a tuple of (h,w) and returns a tuple of (h,w)
        """
        
        if type(h_w) is not tuple:
            h_w = (h_w, h_w)
        
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        
        if type(stride) is not tuple:
            stride = (stride, stride)
        
        if type(pad) is not tuple:
            pad = (pad, pad)
        
        h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
        w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
        
        return h, w





HAS_DISPLAY = True
DEBUG = False
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


def get_entry_point():
    return 'PPOAgent'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


class RlEvaluator(MapAgent):
    def setup(self, path_to_conf_file, path_to_nn):
        super().setup(path_to_conf_file)

        self.car_im_height = 300
        self.car_im_width = 300

        self.amount_images = 3

        self.final_output = [0]*self.amount_images

        print('Running Rl Evaluator')

        #self.save_path = None
        """
        if path_to_conf_file:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(path_to_conf_file) / string
            self.save_path.mkdir(exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'rgb_left').mkdir()
            (self.save_path / 'rgb_right').mkdir()
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'measurements').mkdir()
        """
        self.load_policy_nn(path_to_nn)


    def load_policy_nn(path_to_nn):

        assert(0==1, 'Implement this, this is a virtual method')


    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 2.5, 'y': 0, 'z': 0.7,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 300, 'height': 300, 'fov': 160,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.5, 'y': 0.0, 'z': 0.7,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                    'width': 300, 'height': 300, 'fov': 160,
                    'id': 'rgbback'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 100,
                    'roll': 0.0, 'pitch': 270.0, 'yaw': 0.0,
                    'width': 300, 'height': 300, 'fov': 50,
                    'id': 'minimap'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]



    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        if self.step % 100 == 0:
            index = (self.step // 100) % len(WEATHERS)
            self._world.set_weather(WEATHERS[index])


        data = self.tick(input_data)
        


        #topdown = data['topdown']
        #rgb = np.hstack((data['rgb_left'], data['rgb'], data['rgb_right']))

        #gps = self._get_position(data)

        #near_node, near_command = self._waypoint_planner.run_step(gps)
        #far_node, far_command = self._command_planner.run_step(gps)

        """
        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _rgb = Image.fromarray(rgb)
        _draw = ImageDraw.Draw(_topdown)

        _topdown.thumbnail((256, 256))
        _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))

        _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
        _draw = ImageDraw.Draw(_combined)

        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data, _draw)

        _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
        _draw.text((5, 30), 'Steer: %.3f' % steer)
        _draw.text((5, 50), 'Throttle: %.3f' % throttle)
        _draw.text((5, 70), 'Brake: %s' % brake)

        if HAS_DISPLAY:
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        """

        throttle, steer = self.get_controls_nn(data)

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(0.0)

        """
        if self.step % 10 == 0:
            self.save(far_node, near_command, steer, throttle, brake, target_speed, data)
        """
    
        return control


    def get_controls_nn(self, data):

        assert(True, 'Implement this, this is a virtual method')


    def save(self, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // 10

        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': target_speed,
                'x_command': far_node[0],
                'y_command': far_node[1],
                'command': near_command.value,
                'steer': steer,
                'throttle': throttle,
                'brake': brake,
                }

        (self.save_path / 'measurements' / ('%04d.json' % frame)).write_text(str(data))

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / 'rgb_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['topdown']).save(self.save_path / 'topdown' / ('%04d.png' % frame))



    def tick(self, input_data):
        #self._actors = self._world.get_actors()
        #self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))

        #topdown = input_data['map'][1][:, :, 2]
        #topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)

        #result = super().tick(input_data)

        

        self.step += 1

        np_rgb_front = input_data['rgb'][1][:, :, :3]

        np_rgb_back = input_data['rgbback'][1][:, :, :3]

        np_rgb_minimap = input_data['minimap'][1][:, :, :3]

        result = np.hstack((np_rgb_front, np_rgb_back))
        result = np.hstack((result, np_rgb_minimap))

        #result['topdown'] = topdown

        render = True
        if render:
            cv2.imshow('rl_agent_sensors', result)
            cv2.waitKey(1)

        speed = input_data['speed'][1]['speed']

        aditional_aug = np.zeros((2, 6))

        aditional_aug[1][0] = speed

        return [result.astype(np.float32)/255, aditional_aug]

    def process_img(self, image, numb):
       
        i = np.array(image.raw_data)

        i2 = i.reshape((self.car_im_height, self.car_im_width, 4))

        i3 = i2[:, :, :3]
        
        self.final_output[numb] = i3


    def make_final_output(self):
        
        final_img = self.final_output[0]

        for image in range(1, len(self.final_output)):
            final_img = np.hstack((final_img, self.final_output[image]))

        """
        if self.sensor_img_save:
            im = Image.fromarray(final_img)
            im.save('images/' + str(time.time()).replace('.', '') + '.png')
            
        if self.show_cam:
            pass
            #
            #cv2.imshow("CarlaGymEnv Cam", final_img)
            #cv2.waitKey(1)
        """

        #final_img = final_img.transpose((-1, 0, 1)).astype(np.float32)/255 # Why this

        final_img = final_img.astype(np.float32)/255

        #return [final_img, np.asarray(self.last_ddest)]
        #return [final_img, self.additional_obs()]
        return final_img

    def additional_obs(self):
        additional_obs = np.random.rand(len(self.addtionals_obs_callbacks), 6).astype('float32')

        for i in range(len(self.addtionals_obs_callbacks)):
            to_call = self.addtionals_obs_callbacks[i]
            additional_obs[0] = to_call()

        #print(additional_obs)

        return additional_obs


class PPOAgent(RlEvaluator):

    def setup(self, path_to_conf_file):

        super().setup(path_to_conf_file, '/home/gonvas/Programming/2020_CARLA_challenge/leaderboard/team_code/ppo_nn_1000.tar')


    def load_policy_nn(self, path_to_nn):

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            checkpoint = torch.load(path_to_nn)

            ppo_net = Policy(((300, 900), 3), 9, 12).to(self.device).float()
            ppo_net.load_state_dict(checkpoint['model_state_dict'])

            self.policy = ppo_net


    def get_controls_nn(self, data, steer_amt=1.1):


        obs = (torch.Tensor(data[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(self.device).float(), torch.Tensor(data[1]).to(self.device).float())

        dist, val = self.policy(obs)

        action = dist.sample().item()

        action = int(action/3), action%3

        thrt_action, steer_action = action
        #Discrete(3) -> 0, 1, 2 -> transform to -1, 0, 1
        thrt_action -= 1
        steer_action -= 1

        #return 1.0, 1.0
        return thrt_action*1.0, steer_action*steer_amt



