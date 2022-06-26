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
body_width = cfg['body_width']
body_height = cfg['body_height']

BODY_POLY = [(-body_width / 2, 120), (-body_width / 2, -120),
             (+body_width / 2, -120), (+body_width / 2, 120)]
WHEELPOS = [(-body_width, +80), (+body_width, +80), (-body_width, -80),
            (+body_width, -80)]
wheel_height = 30
wheel_width = 24


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
        self.body.color = cfg['body_color']
        self.wheels = list()
        self.energy_expended = 0.0
        WHEEL_POLY = [
            (-wheel_width, +wheel_height),
            (+wheel_width, +wheel_height),
            (+wheel_width, -wheel_height),
            (-wheel_width, -wheel_height),
        ]
        for wx, wy in WHEELPOS:
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * self.size, init_y + wy * self.size),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x * self.size, y * self.size)
                                                 for x, y in WHEEL_POLY]),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = wheel_height * self.size
            w.color = cfg['wheel_color']
            w.acc = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.body,
                bodyB=w,
                localAnchorA=(wx * self.size, wy * self.size),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * self.size * self.size,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.render_list = self.wheels + [self.body]

    def accelerate(self, acc: float) -> None:
        acc = np.clip(acc, 0, 1)
        for w in self.wheels:
            diff = acc - w.acc
            w.acc += diff

    def decelerate(self, d: float) -> None:
        for w in self.wheels:
            w.brake = d

    # def step(self, dt) -> None:
    #     for w in self.wheels:
    #         pass

    def render(self, surface, zoom: float, translation, angle: float) -> None:
        for obj in self.render_list:
            for fix in obj.fixtures:
                trans = fix.body.transform
                path = [trans * v for v in fix.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [(
                    coords[0] * zoom + translation[0],
                    coords[1] * zoom + translation[1],
                ) for coords in path]
                color = [c for c in obj.color]

                pygame.draw.polygon(surface, color=color, points=path)

    def destroy(self) -> None:
        self.world.DestroyBody(self.body)
        self.body = None
