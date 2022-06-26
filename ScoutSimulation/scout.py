# Influenced heavily by the fantastic Car class in car_dynamics.py

import yaml
import math
import numpy as np

from gym.error import DependencyNotInstalled

import pygame.draw

try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
except ImportError:
    raise DependencyNotInstalled(
        "box2D is not installed, run `pip install gym[box2d]`")

with open('config.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg = cfg['Scout']

BODY_POLY = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]


class Scout:

    def __init__(self, world, init_angle: float, init_x: float,
                 init_y: float) -> None:
        self.world = world
        self.size = cfg['size']
        self.body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(shape=polygonShape(vertices=[(x * self.size,
                                                         y * self.size)
                                                        for x, y in BODY_POLY]),
                           density=1.0)
            ])
        self.body.color = cfg['color']
        self.wheels = list()
        self.energy_expended = 0.0
        self.render_list = [self.body]

    def render(self, surface, zoom: float, translation, angle: float) -> None:
        for obj in self.render_list:
            for fix in obj.fixtures:
                trans = fix.body.transform
                path = [trans * v for v in fix.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in path
                ]
                color = [c for c in obj.color]

                pygame.draw.polygon(surface, color=color, points=path)