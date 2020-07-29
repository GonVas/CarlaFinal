import glob
import os
import sys
import random
import time
import numpy as np
from numpy.linalg import inv
import math
from math import pi
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
from carla import TrafficLightState as tls

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





COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)


COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

PIXELS_PER_METER = 10

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0

PIXELS_AHEAD_VEHICLE = 150


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class Util(object):

    @staticmethod
    def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def get_bounding_box(actor):
        bb = actor.trigger_volume.extent
        corners = [carla.Location(x=-bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners



class ModuleHUD (object):

    def __init__(self, name, width, height):
        self.name = name
        self.dim = (width, height)
        self._init_hud_params()
        self._init_data_params()

    def start(self):
        pass

    def _init_hud_params(self):
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._header_font = pygame.font.SysFont('Arial', 14, True)
        self.help = HelpText(pygame.font.Font(mono, 24), *self.dim)
        self._notifications = FadingText(
            pygame.font.Font(pygame.font.get_default_font(), 20),
            (self.dim[0], 40), (0, self.dim[1] - 40))

    def _init_data_params(self):
        self.show_info = True
        self.show_actor_ids = False
        self._info_text = {}

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def tick(self, clock):
        self._notifications.tick(clock)

    def add_info(self, module_name, info):
        self._info_text[module_name] = info

    def render_vehicles_ids(self, vehicle_id_surface, list_actors, world_to_pixel, hero_actor, hero_transform):
        vehicle_id_surface.fill(COLOR_BLACK)
        if self.show_actor_ids:
            vehicle_id_surface.set_alpha(150)
            for actor in list_actors:
                x, y = world_to_pixel(actor[1].location)

                angle = 0
                if hero_actor is not None:
                    angle = -hero_transform.rotation.yaw - 90

                color = COLOR_SKY_BLUE_0
                if int(actor[0].attributes['number_of_wheels']) == 2:
                    color = COLOR_CHOCOLATE_0
                if actor[0].attributes['role_name'] == 'hero':
                    color = COLOR_CHAMELEON_0

                font_surface = self._header_font.render(str(actor[0].id), True, color)
                rotated_font_surface = pygame.transform.rotate(font_surface, angle)
                rect = rotated_font_surface.get_rect(center=(x, y))
                vehicle_id_surface.blit(rotated_font_surface, rect)

        return vehicle_id_surface

    def render(self, display):
        if self.show_info:
            info_surface = pygame.Surface((240, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            i = 0
            for module_name, module_info in self._info_text.items():
                if not module_info:
                    continue
                surface = self._header_font.render(module_name, True, COLOR_ALUMINIUM_0).convert_alpha()
                display.blit(surface, (8 + bar_width / 2, 18 * i + v_offset))
                v_offset += 12
                i += 1
                for item in module_info:
                    if v_offset + 18 > self.dim[1]:
                        break
                    if isinstance(item, list):
                        if len(item) > 1:
                            points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                            pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                        item = None
                    elif isinstance(item, tuple):
                        if isinstance(item[1], bool):
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                            pygame.draw.rect(display, COLOR_ALUMINIUM_0, rect, 0 if item[1] else 1)
                        else:
                            rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                            pygame.draw.rect(display, COLOR_ALUMINIUM_0, rect_border, 1)
                            f = (item[1] - item[2]) / (item[3] - item[2])
                            if item[2] < 0.0:
                                rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                            else:
                                rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                            pygame.draw.rect(display, COLOR_ALUMINIUM_0, rect)
                        item = item[0]
                    if item:  # At this point has to be a str.
                        surface = self._font_mono.render(item, True, COLOR_ALUMINIUM_0).convert_alpha()
                        display.blit(surface, (8, 18 * i + v_offset))
                    v_offset += 18
                v_offset += 24
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- TrafficLightSurfaces ------------------------------------------------------
# ==============================================================================


class TrafficLightSurfaces(object):
    """Holds the surfaces (scaled and rotated) for painting traffic lights"""

    def __init__(self):
        def make_surface(tl):
            w = 40
            surface = pygame.Surface((w, 3 * w), pygame.SRCALPHA, 32)
            surface.fill(COLOR_ALUMINIUM_5 if tl != 'h' else COLOR_ORANGE_2)
            if tl != 'h':
                hw = int(w / 2)
                off = COLOR_ALUMINIUM_4
                red = COLOR_SCARLET_RED_0
                yellow = COLOR_BUTTER_0
                green = COLOR_CHAMELEON_0
                pygame.draw.circle(surface, red if tl == tls.Red else off, (hw, hw), int(0.4 * w))
                pygame.draw.circle(surface, yellow if tl == tls.Yellow else off, (hw, w + hw), int(0.4 * w))
                pygame.draw.circle(surface, green if tl == tls.Green else off, (hw, 2 * w + hw), int(0.4 * w))
            return pygame.transform.smoothscale(surface, (15, 45) if tl != 'h' else (19, 49))
        self._original_surfaces = {
            'h': make_surface('h'),
            tls.Red: make_surface(tls.Red),
            tls.Yellow: make_surface(tls.Yellow),
            tls.Green: make_surface(tls.Green),
            tls.Off: make_surface(tls.Off),
            tls.Unknown: make_surface(tls.Unknown)
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle, scale):
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(surface, angle, scale)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter, show_triggers, show_connections, show_spawn_points):
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0
        self.show_triggers = show_triggers
        self.show_connections = show_connections
        self.show_spawn_points = show_spawn_points

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.draw_road_map(self.big_map_surface, carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)
        self.surface = self.big_map_surface

    def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        map_surface.fill(COLOR_ALUMINIUM_4)
        precision = 0.05

        def lane_marking_color_to_tango(lane_marking_color):
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def draw_solid_line(surface, color, closed, points, width):
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(surface, color, closed, points, width):
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            margin = 0.20
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
                marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
                return [(lane_marking_type, lane_marking_color, marking_1)]
            elif lane_marking_type == carla.LaneMarkingType.SolidBroken or lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
                marking_2 = [world_to_pixel(lateral_shift(w.transform,
                                                          sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                marking = [world_to_pixel(lateral_shift(w.transform,
                                                        sign * (w.lane_width * 0.5 - margin))) for w in waypoints]
                return [(carla.LaneMarkingType.Broken, lane_marking_color, marking)]
            elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                marking = [world_to_pixel(lateral_shift(w.transform,
                                                        sign * ((w.lane_width * 0.5) - margin))) for w in waypoints]
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

        def draw_lane_marking(surface, waypoints, is_left):
            sign = -1 if is_left else 1
            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE

            marking_color = carla.LaneMarkingColor.Other
            previous_marking_color = carla.LaneMarkingColor.Other

            waypoints_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign)
                    current_lane_marking = marking_type

                    for marking in markings:
                        waypoints_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]

                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # Add last marking
            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign)
            for marking in last_markings:
                waypoints_list.append(marking)

            for markings in waypoints_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        start, end]], 4)
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        left, start, right]], 4)

        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

            # draw bounding box
            if self.show_triggers:
                corners = Util.get_bounding_box(actor)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, trigger_color, True, corners, 2)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            for waypoint in topology:
                # if waypoint.road_id == 150 or waypoint.road_id == 16:
                waypoints = [waypoint]

                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break

                # Draw Road
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Draw Shoulders and Parkings
                PARKING_COLOR = COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5

                final_color = SHOULDER_COLOR

                # Draw Right
                shoulder = []
                for w in waypoints:
                    r = w.get_right_lane()
                    if r is not None and (
                            r.lane_type == carla.LaneType.Shoulder or r.lane_type == carla.LaneType.Parking):
                        if r.lane_type == carla.LaneType.Parking:
                            final_color = PARKING_COLOR
                        shoulder.append(r)

                shoulder_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in shoulder]
                shoulder_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in shoulder]

                polygon = shoulder_left_side + [x for x in reversed(shoulder_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, final_color, polygon, 5)
                    pygame.draw.polygon(map_surface, final_color, polygon)

                draw_lane_marking(
                    map_surface,
                    shoulder,
                    False)

                # Draw Left
                shoulder = []
                for w in waypoints:
                    r = w.get_left_lane()
                    if r is not None and (
                            r.lane_type == carla.LaneType.Shoulder or r.lane_type == carla.LaneType.Parking):
                        if r.lane_type == carla.LaneType.Parking:
                            final_color = PARKING_COLOR
                        shoulder.append(r)

                shoulder_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in shoulder]
                shoulder_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in shoulder]

                polygon = shoulder_left_side + [x for x in reversed(shoulder_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, final_color, polygon, 5)
                    pygame.draw.polygon(map_surface, final_color, polygon)

                draw_lane_marking(
                    map_surface,
                    shoulder,
                    True)

                # Draw Lane Markings and Arrows
                if not waypoint.is_intersection:
                    draw_lane_marking(
                        map_surface,
                        waypoints,
                        True)
                    draw_lane_marking(
                        map_surface,
                        waypoints,
                        False)
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)
        draw_topology(topology, 1)

        if self.show_spawn_points:
            for sp in carla_map.get_spawn_points():
                draw_arrow(map_surface, sp, color=COLOR_CHOCOLATE_0)

        if self.show_connections:
            dist = 1.5
            to_pixel = lambda wp: world_to_pixel(wp.transform.location)
            for wp in carla_map.generate_waypoints(dist):
                col = (0, 255, 255) if wp.is_intersection else (0, 255, 0)
                for nxt in wp.next(dist):
                    pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(nxt), 2)
                if wp.lane_change & carla.LaneChange.Right:
                    r = wp.get_right_lane()
                    if r and r.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(r), 2)
                if wp.lane_change & carla.LaneChange.Left:
                    l = wp.get_left_lane()
                    if l and l.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(l), 2)

        actors = carla_world.get_actors()

        # Draw Traffic Signs
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)

        stops = [actor for actor in actors if 'stop' in actor.type_id]
        yields = [actor for actor in actors if 'yield' in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

        for ts_stop in stops:
            draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        for ts_yield in yields:
            draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))


class ModuleWorld(object):
    def __init__(self, name, timeout):
        self.client = None
        self.name = name
        #self.args = args
        self.timeout = timeout
        self.server_fps = 0.0
        self.simulation_time = 0
        self.server_clock = pygame.time.Clock()

        # World data
        self.world = None
        self.town_map = None
        self.actors_with_transforms = []
        # Store necessary modules
        #self.module_hud = None
        self.module_hud_dim = [1280, 720]
        self.module_input = None

        self.surface_size = [0, 0]
        self.prev_scaled_size = 0
        self.scaled_size = 0
        # Hero actor
        self.hero_actor = None
        self.spawned_hero = None
        self.hero_transform = None

        self.scale_offset = [0, 0]

        self.vehicle_id_surface = None
        self.result_surface = None

        self.traffic_light_surfaces = TrafficLightSurfaces()
        self.affected_traffic_light = None

        # Map info
        self.map_image = None
        self.border_round_surface = None
        self.original_surface_size = None
        self.hero_surface = None
        self.actors_surface = None

    def _get_data_from_carla(self):

        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(4)


        world = self.client.get_world()

        town_map = world.get_map()
        return (world, town_map)


    def start(self, hero_actor):
        self.world, self.town_map = self._get_data_from_carla()

        # Create Surfaces
        self.map_image = MapImage(
            carla_world=self.world,
            carla_map=self.town_map,
            pixels_per_meter=PIXELS_PER_METER,
            show_triggers=True,
            show_connections=True,
            show_spawn_points=True)

        # Store necessary modules
        #self.module_hud = module_manager.get_module(MODULE_HUD)
        #self.module_input = module_manager.get_module(MODULE_INPUT)



        self.original_surface_size = min(self.module_hud_dim[0], self.module_hud_dim[1])
        self.surface_size = self.map_image.big_map_surface.get_width()

        self.scaled_size = int(self.surface_size)
        self.prev_scaled_size = int(self.surface_size)

        # Render Actors
        self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)

        self.vehicle_id_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.vehicle_id_surface.set_colorkey(COLOR_BLACK)

        self.border_round_surface = pygame.Surface(self.module_hud_dim, pygame.SRCALPHA, 32).convert()
        self.border_round_surface.set_colorkey(COLOR_WHITE)
        self.border_round_surface.fill(COLOR_BLACK)

        center_offset = (int(self.module_hud_dim[0] / 2), int(self.module_hud_dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_ALUMINIUM_1, center_offset, int(self.module_hud_dim[1] / 2))
        pygame.draw.circle(self.border_round_surface, COLOR_WHITE, center_offset, int((self.module_hud_dim[1] - 8) / 2))

        scaled_original_size = self.original_surface_size * (1.0 / 0.9)
        self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()

        self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)

        # Start hero mode by default
        #self.select_hero_actor()
        #self.hero_actor.set_autopilot(False)
        #self.module_input.wheel_offset = HERO_DEFAULT_SCALE
        #self.module_input.control = carla.VehicleControl()

        self.hero_actor = hero_actor

        weak_self = weakref.ref(self)
        self.world.on_tick(lambda timestamp: ModuleWorld.on_world_tick(weak_self, timestamp))

    def select_hero_actor(self):
        hero_vehicles = [actor for actor in self.world.get_actors(
        ) if 'vehicle' in actor.type_id and actor.attributes['role_name'] == 'hero']
        if len(hero_vehicles) > 0:
            self.hero_actor = random.choice(hero_vehicles)
            self.hero_transform = self.hero_actor.get_transform()
        else:
            pass
            #self._spawn_hero()

    def _spawn_hero(self):
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self.args.filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        while self.hero_actor is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.hero_actor = self.world.try_spawn_actor(blueprint, spawn_point)
        self.hero_transform = self.hero_actor.get_transform()

        # Save it in order to destroy it when closing program
        self.spawned_hero = self.hero_actor

    def tick(self, clock):
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        #self.update_hud_info(clock)

    def update_hud_info(self, clock):
        hero_mode_text = []
        if self.hero_actor is not None:
            hero_speed = self.hero_actor.get_velocity()
            hero_speed_text = 3.6 * math.sqrt(hero_speed.x ** 2 + hero_speed.y ** 2 + hero_speed.z ** 2)

            affected_traffic_light_text = 'None'
            if self.affected_traffic_light is not None:
                state = self.affected_traffic_light.state
                if state == carla.TrafficLightState.Green:
                    affected_traffic_light_text = 'GREEN'
                elif state == carla.TrafficLightState.Yellow:
                    affected_traffic_light_text = 'YELLOW'
                else:
                    affected_traffic_light_text = 'RED'

            affected_speed_limit_text = self.hero_actor.get_speed_limit()

            hero_mode_text = [
                'Hero Mode:                 ON',
                'Hero ID:              %7d' % self.hero_actor.id,
                'Hero Vehicle:  %14s' % get_actor_display_name(self.hero_actor, truncate=14),
                'Hero Speed:          %3d km/h' % hero_speed_text,
                'Hero Affected by:',
                '  Traffic Light: %12s' % affected_traffic_light_text,
                '  Speed Limit:       %3d km/h' % affected_speed_limit_text
            ]
        else:
            hero_mode_text = ['Hero Mode:                OFF']

        self.server_fps = self.server_clock.get_fps()
        self.server_fps = 'inf' if self.server_fps == float('inf') else round(self.server_fps)
        module_info_text = [
            'Server:  % 16s FPS' % self.server_fps,
            'Client:  % 16s FPS' % round(clock.get_fps()),
            'Simulation Time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            'Map Name:          %10s' % self.town_map.name,
        ]

        module_info_text = module_info_text
        module_hud = module_manager.get_module(MODULE_HUD)
        module_hud.add_info(self.name, module_info_text)
        module_hud.add_info('HERO', hero_mode_text)

    @staticmethod
    def on_world_tick(weak_self, timestamp):
        self = weak_self()
        if not self:
            return

        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.simulation_time = timestamp.elapsed_seconds

    def _split_actors(self):
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker' in actor.type_id:
                walkers.append(actor_with_transform)

        info_text = []
        if self.hero_actor is not None and len(vehicles) > 1:
            location = self.hero_transform.location
            vehicle_list = [x[0] for x in vehicles if x[0].id != self.hero_actor.id]

            def distance(v): return location.distance(v.get_location())
            for n, vehicle in enumerate(sorted(vehicle_list, key=distance)):
                if n > 15:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                info_text.append('% 5d %s' % (vehicle.id, vehicle_type))
        #module_manager.get_module(MODULE_HUD).add_info(
        #    'NEARBY VEHICLES',
        #    info_text)

        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_traffic_lights(self, surface, list_tl, world_to_pixel):
        self.affected_traffic_light = None

        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)


            corners = Util.get_bounding_box(tl)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(surface, COLOR_BUTTER_1, True, corners, 2)

            if self.hero_actor is not None:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                tl_t = tl.get_transform()

                transformed_tv = tl_t.transform(tl.trigger_volume.location)
                hero_location = self.hero_actor.get_location()
                d = hero_location.distance(transformed_tv)
                s = Util.length(tl.trigger_volume.extent) + Util.length(self.hero_actor.bounding_box.extent)
                if (d <= s):
                    # Highlight traffic light
                    self.affected_traffic_light = tl
                    srf = self.traffic_light_surfaces.surfaces['h']
                    surface.blit(srf, srf.get_rect(center=pos))

            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))

    def _render_speed_limits(self, surface, list_sl, world_to_pixel, world_to_pixel_width):

        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)

        for sl in list_sl:

            x, y = world_to_pixel(sl.get_location())

            # Render speed limit
            white_circle_radius = int(radius * 0.75)

            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius)

            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)


            corners = Util.get_bounding_box(sl)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(surface, COLOR_PLUM_2, True, corners, 2)

            # Blit
            if self.hero_actor is not None:
                # Rotate font surface with respect to hero vehicle front
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)

            else:
                surface.blit(font_surface, (x - radius / 2, y - radius / 2))

    def _render_walkers(self, surface, list_w, world_to_pixel):
        for w in list_w:
            color = COLOR_PLUM_0

            # Compute bounding box points
            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y)]

            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v, world_to_pixel):

        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v[0].attributes['role_name'] == 'hero':
                color = COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.lines(surface, color, False, corners, int(math.ceil(4.0 * self.map_image.scale)))


    def render_actors(self, surface, vehicles, traffic_lights, speed_limits, walkers):
        # Static actors
        self._render_traffic_lights(surface, [tl[0] for tl in traffic_lights], self.map_image.world_to_pixel)
        self._render_speed_limits(surface, [sl[0] for sl in speed_limits], self.map_image.world_to_pixel,
                                  self.map_image.world_to_pixel_width)

        # Dynamic actors
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)

    def clip_surfaces(self, clipping_rect):
        self.actors_surface.set_clip(clipping_rect)
        self.vehicle_id_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)


    def render(self, display):
        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(COLOR_BLACK)
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()

        scale_factor = 1 #self.module_input.wheel_offset
        #self.scaled_size = int(self.map_image.width * scale_factor)
        #if self.scaled_size != self.prev_scaled_size:
        #    self._compute_scale(scale_factor)

        # Render Actors

        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(
            self.actors_surface,
            vehicles,
            traffic_lights,
            speed_limits,
            walkers)

        # Render Ids
        #self.module_hud.render_vehicles_ids(self.vehicle_id_surface, vehicles,
        #                                    self.map_image.world_to_pixel, self.hero_actor, self.hero_transform)

        # Blit surfaces
        surfaces = ((self.map_image.surface, (0, 0)),
                    (self.actors_surface, (0, 0)),
                    (self.vehicle_id_surface, (0, 0)),
                    )

        angle = 0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90.0
        self.traffic_light_surfaces.rotozoom(-angle, self.map_image.scale)

        center_offset = (0, 0)
        if self.hero_actor is not None:
            print('wefwefwefwefwe')
            hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
            hero_front = self.hero_transform.get_forward_vector()
            translation_offset = (
                hero_location_screen[0] -
                self.hero_surface.get_width() /
                2 +
                hero_front.x *
                PIXELS_AHEAD_VEHICLE,
                (hero_location_screen[1] -
                 self.hero_surface.get_height() /
                 2 +
                 hero_front.y *
                 PIXELS_AHEAD_VEHICLE))

            # Apply clipping rect
            clipping_rect = pygame.Rect(translation_offset[0],
                                        translation_offset[1],
                                        self.hero_surface.get_width(),
                                        self.hero_surface.get_height())
            self.clip_surfaces(clipping_rect)

            Util.blits(self.result_surface, surfaces)

            self.border_round_surface.set_clip(clipping_rect)

            self.hero_surface.fill(COLOR_ALUMINIUM_4)
            self.hero_surface.blit(self.result_surface, (-translation_offset[0],
                                                         -translation_offset[1]))

            rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()

            center = (display.get_width() / 2, display.get_height() / 2)
            rotation_pivot = rotated_result_surface.get_rect(center=center)
            display.blit(rotated_result_surface, rotation_pivot)

            display.blit(self.border_round_surface, (0, 0))

        """
        else:
            # Translation offset
            print(self.module_input.mouse_offset[0])
            translation_offset = (self.module_input.mouse_offset[0] * scale_factor + self.scale_offset[0],
                                  self.module_input.mouse_offset[1] * scale_factor + self.scale_offset[1])
            center_offset = (abs(display.get_width() - self.surface_size) / 2 * scale_factor, 0)

            # Apply clipping rect
            clipping_rect = pygame.Rect(-translation_offset[0] - center_offset[0], -translation_offset[1],
                                        self.module_hud_dim[0], self.module_hud_dim[1])
            self.clip_surfaces(clipping_rect)
            Util.blits(self.result_surface, surfaces)

            display.blit(self.result_surface, (translation_offset[0] + center_offset[0],
                                               translation_offset[1]))
        """

    def destroy(self):
        if self.spawned_hero is not None:
            self.spawned_hero.destroy()




















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
                    display2d=True,
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
        self.model_3.set_attribute('role_name', 'hero')

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

        self.display2d = display2d

        if(display2d == True):
            pygame.init()
            self.minimap_2d_display = pygame.display.set_mode(
                (1280, 720),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            self.minimap_clock = pygame.time.Clock()
            pygame.display.flip()
            #import pudb; pudb.set_trace()
            self.world_module = ModuleWorld('WORLD', timeout=2.0)

            self.world_module.start(self.player)
            self.minimap_2d_display.fill(COLOR_ALUMINIUM_4)
            #module_manager.tick(clock)

            #img_numpy = pygame.surfarray.array3d(display)

            #cv2.imshow('Pygame Surface', img_numpy)
            #cv2.waitKey(1)
            #pil_string_image = pygame.image.tostring(rImage, "RGBA",False)
            #pil_image = Image.fromstring("RGBA",(660,660),pil_string_image)

            pygame.display.flip()          





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



    def render_2d_minimap(self):

        self.minimap_2d_display.fill(COLOR_ALUMINIUM_4)
        #module_manager.tick(clock)
        self.minimap_clock.tick()
        self.world_module.tick(self.minimap_clock)
        self.world_module.render(self.minimap_2d_display)
        #module_manager.render(display)

        #import pudb; pudb.set_trace()

        #img_numpy = pygame.surfarray.array3d(display)
        #img_numpy = pygame.surfarray.array3d(self.minimap_2d_display)

        #cv2.imshow('Pygame Surface', img_numpy)
        #cv2.waitKey(1)

        #cv2.imshow('Pygame Surface', img_numpy)
        #cv2.waitKey(1)
        #pil_string_image = pygame.image.tostring(rImage, "RGBA",False)
        #pil_image = Image.fromstring("RGBA",(660,660),pil_string_image)
        #pygame.display.flip()


    def add_sensors(self, sensors):
        self.static_index = 0
        for sensor in sensors:
            if(sensor == 'rgbcam1'):
                rgb_cam1 = self.blueprint_library.find('sensor.camera.rgb')
                rgb_cam1.set_attribute("image_size_x", f"{self.car_im_width}")
                rgb_cam1.set_attribute("image_size_y", f"{self.car_im_height}")
                rgb_cam1.set_attribute("fov", f"100")

                
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
                rgb_cam2.set_attribute("fov", f"100")

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
                rgb_cam3.set_attribute("fov", f"100")

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
                rgbback.set_attribute("fov", f"100")

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


        self.front_camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.minimap_camera_transform = carla.Transform(location=carla.Location(x=0.0, y=0.0, z=100.0), rotation=carla.Rotation(pitch=270.0, yaw=0.0, roll=0.0))

        """
        lidar = self.blueprint_library.find('sensor.lidar.ray_cast')
        #lidar.set_attribute("image_size_x", f"{self.car_im_width}")
        #lidar.set_attribute("image_size_y", f"{self.car_im_height}")
        #lidar.set_attribute("fov", f"50")

        lidar.set_attribute('channels',str(32))
        lidar.set_attribute('points_per_second',str(100000))
        lidar.set_attribute('rotation_frequency',str(10))
        lidar.set_attribute('range',str(50))
        lidar.set_attribute('upper_fov',str(10))
        lidar.set_attribute('lower_fov',str(-30))
        lidar_location = carla.Location(0,0,2.5)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)

        self.lidar_transform = carla.Transform(lidar_location,lidar_rotation)

        lidar = self.world.spawn_actor(lidar, lidar_transform, attach_to=self.vehicle)


        #import pudb; pudb.set_trace()

        #what_index4 = int(self.static_index) + 1 - 1
        lidar.listen(lambda data: self.process_lidar(data))
        self.actor_list.append((lidar))
        """
        #def get_unreal_transform(self): 
            #to_unreal_transform = Transform(Rotation(yaw=90), Scale(z=-1))
            #return self.get_transform() * to_unreal_transform

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


    def process_lidar(self, data):

        #img_h, img_w, height, width = 300
        """

        WINDOW_WIDTH = 200
        WINDOW_HEIGHT = 160

        WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
        WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        k[0, 0] = k[1, 1] = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))

        lidar_data = np.asarray(data.raw_data, np.int32).reshape(-1, 4)

        rot = [0, 270, 0]
        # pitch (Y) - yaw (Z) - roll (X)
        """



        

        #rot = [0, 3*pi/2, 0]


        #c = [130, 125, 250]   

        #print(data.transform.location)
        #print(np.asarray(data.raw_data).reshape(-1, 4)[:10])

        #data_ = np.asarray(data.raw_data).reshape(-1, 4)

        #print('Mean_x : {}, Max_x:{}, Min_y:{}'.format(data_[:, 0].mean(), data_[:, 0].max(), data_[:, 0].min()))

        #c = [200, -100, data.transform.location.z + 19]

        #[130, 200, 250]

        #c = [0.5, 0.5, 0.5]

        #print([self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y, data.transform.location.z + 18])

        #s = [300, 300]

        #r = [300, 300, 1]

        #e = [3, 3, 3]
        #blank_image = np.zeros((300, 300, 3), np.uint32)
        #cl_point = 10000000

        #eps = 1e-6

        #import pudb; pudb.set_trace()
        """
        for idx, point in enumerate(np.asarray(data.raw_data).reshape(-1, 4)):
            
            #point_no_depth = lidar_data[:, [0, 1, 2]]
            #point_depth = lidar_data[:, 3]
            #if(idx == 4826):
            #    import pudb; pudb.set_trace()

            a = point[:3] / 255
            point_depth = point[3]

            m_1 = np.asarray([[1, 0, 0], [0, np.cos(rot[0]), np.sin(rot[0])], [0, -np.sin(rot[0]), np.cos(rot[0])]])
            m_2 = np.asarray([[np.cos(rot[1]), 0, -np.sin(rot[1])], [0, 1, 0], [np.sin(rot[1]), 0, np.cos(rot[1])]])
            m_3 = np.asarray([[np.cos(rot[2]), np.sin(rot[2]), 0], [-np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
            vec = a - c

            d_vec = m_1 @ m_2 @ m_3 @ vec

            bx = int((d_vec[0] * s[0]) / (d_vec[2]*r[0] + eps)    *   r[2])
            by = int((d_vec[1] * s[1]) / (d_vec[2]*r[1] + eps)    *   r[2])

            if(bx > 0 and bx < 300 and by > 0 and by < 300):
                #blank_image[bx, by][1] =  np.clip(point_depth + blank_image[bx, by][1], 0, cl_point)
                #blank_image[bx, by][1] =  max(point_depth,  blank_image[bx, by][1])
                blank_image[bx, by][1] += point_depth
        """

        hud_dim = [300, 300]

        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        #points[:, [2]] *= -1

        lidar_data = np.array(points[:, :2])
        lidar_data *= min(hud_dim) / 100.0
        lidar_data += (0.5 * hud_dim[0], 0.5 * hud_dim[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        #lidar_data[2] *= -1
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (hud_dim[0], hud_dim[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype = int)

#for idx, point in enumerate(lidar_data):
    #import pudb; pudb.set_trace()
#    lidar_img[point[0]][point[1]][1] += points[idx][3]*10
    #lidar_img[point.T] = (0, points[idx][3] + lidar_img[point.T][1], 0)


        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)


        #for idx, point in enumerate(np.asarray(data.raw_data).reshape(-1, 4)):
        #    point_depth = point[3]
        #    blank_image[point[1]][point[2]][1]+= point_depth*0.001

        #cv2.imwrite('./depth_test_{}.png'.format(15), lidar_img) 
        print('Shwoing')
        cv2.imshow('Proj_x_test', lidar_img.astype(np.uint8))
        cv2.waitKey(0)
        cv2.waitKey(1)
        print('Shown')





        """
        
        lidar_to_car_transform_matrix = CarEnv.get_matrix(self.lidar_transform)
        camera_to_car_transform_matrix = CarEnv.get_matrix(self.minimap_camera_transform)

        #lidar_data

        for i in range(lidar_data.shape[0]):
            pos_vector = np.array([  [lidar_data[i,0]], [lidar_data[i,1]], [lidar_data[i,2]], [1.0]])
            point_pos = np.dot(lidar_to_car_transform_matrix, pos_vector)
            point_pos = np.dot(inv(camera_to_car_transform_matrix), point_pos)
            pos2d = np.dot(k, point_pos[:3])
            pos2d = np.array([pos2d[0] / pos2d[2], pos2d[1] / pos2d[2], pos2d[2]])
            if pos2d[2] > 0:
                x_2d = WINDOW_WIDTH - pos2d[0]
                y_2d = WINDOW_HEIGHT - pos2d[1]
                if (x_2d >= 0 and x_2d < WINDOW_WIDTH and y_2d >= 0 and y_2d < WINDOW_HEIGHT):
                    print(x_2d, y_2d)
                    #draw_rect(x_2d, y_2d, rgb_image)
        """

        """
        project_matrix = np.asarray([[1,0,0], [0,0,0], [0,0,1]])
        lidar_data = np.asarray(data.raw_data).reshape(-1, 4)

        def f2d(x):
            return project_matrix @ x

        lidar_data_nodepth = lidar_data[:, [0, 1, 2]]
        lidar_just_depth = lidar_data[:, 3]

        
        blank_image = np.zeros((50, 50, 3), np.uint8)
    
        lidar_in_2d = np.asarray(list(map(f2d, lidar_data_nodepth)))
        lidar_in_2d = lidar_in_2d[:, [0, 2]]

        for idx, point in enumerate(lidar_in_2d):

            if(point.max() > 49):
                continue

            if(blank_image[point[0]][point[1]][0] == 0):
                blank_image[point[0]][point[1]] = (lidar_just_depth[idx], 0, 0)
            else:
                blank_image[point[0]][point[1]] = (min(lidar_just_depth[idx], blank_image[point[0]][point[1]][0]), 0, 0)

        
        cv2.imwrite('lidar_proj{}.png'.format(self.global_step_numb), blank_image) 
        #cv2.imshow('Proj', blank_image)
        #cv2.waitKey(1)
        """

        if(self.global_step_numb != 0 and self.global_step_numb % 10 == 0):
            data.save_to_disk('./data/lidar/%.6d.ply' % data.frame)
            with open('lidar_array_{}.npy'.format(self.global_step_numb), 'wb') as f:
                np.save(f, np.asarray(data.raw_data))


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

        if(self.display2d):
            self.world_module.hero_actor = self.vehicle

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
            reward = -0.1
 
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
            return 10, True

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
            info = {'scen_sucess':1, 'scen_metric':10}

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

        if(self.display2d):
            self.render_2d_minimap()

        if(continuos == False):

            if(isinstance(action, int)):
                action = int(action/3), action%3

            thrt_action, steer_action = action
            #Discrete(3) -> 0, 1, 2 -> transform to -1, 0, 1
            thrt_action -= 1
            steer_action -= 1
            if(thrt_action > 0):
                self.vehicle.apply_control(carla.VehicleControl(throttle=thrt_action, steer=steer_action*self.steer_amt))
            else:
                self.vehicle.apply_control(carla.VehicleControl(brake=-thrt_action, steer=steer_action*self.steer_amt))
        else:
            #self.vehicle.apply_control(carla.VehicleControl(throttle=action[0][0], steer=action[1][0]))
            if(action[0].item() > 0):
                self.vehicle.apply_control(carla.VehicleControl(throttle=action[0].item(), steer=action[1].item()))
            else:
                self.vehicle.apply_control(carla.VehicleControl(brake=-action[0].item(), steer=action[1].item()))
        
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



