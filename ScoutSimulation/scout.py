# Influenced heavily by the fantastic Car class in car_dynamics.py

import yaml
import math
import numpy as np

from gym.error import DependencyNotInstalled

import pygame.draw

try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef, wheelJointDef, weldJointDef
    from Box2D.b2 import rayCastOutput, rayCastInput, vec2
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

HEADPOS = (body_width / 4, body_height)
# sensor_x_adj = 5
# sensor_array_poly = [
#     (-body_width / 4, 120),
#     (body_width / 4, 120),
#     (body_width / 4, 100),
#     (-body_width / 4, 100)
# ]

WHEELPOS = [(-body_width, +80), (+body_width, +80), (-body_width, -80),
            (+body_width, -80)]

wheel_height = 30
wheel_width = 24
engine_power = 100000000 * cfg['size'] ** 2


class Scout:

    def __init__(self, world, init_angle: float, init_x: float,
                 init_y: float) -> None:
        self.world = world
        self.size = cfg['size']
        HEAD_POLY = [(-20, 10), (-20, -10), (20, 10), (20, -10)]
        self.angle = init_angle
        self.rotate_flag = False
        self.sensor_array = self.world.CreateDynamicBody(
            position=(init_x + HEADPOS[0] * self.size,
                      init_y + HEADPOS[1] * self.size),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(vertices=[(x * self.size, y * self.size)
                                                 for x, y in HEAD_POLY]),
                    # density=cfg['sensor_array_density'],
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                )
            ])
        self.body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(shape=polygonShape(vertices=[(x * self.size,
                                                         y * self.size)
                                                        for x, y in BODY_POLY]))
                # density=cfg['body_density'])
            ])
        self.raycast = self.world.CreateDynamicBody(
            position=(HEADPOS[0] * self.size, HEADPOS[1] * self.size),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(vertices=[(-0.01,
                                                  0), (0.01,
                                                       0), (0.01,
                                                            10), (-0.01, 10)]),
                    # density=cfg['ray_density'],
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=1e-11,
                ),
                fixtureDef(
                    shape=polygonShape(vertices=[(-0.01,
                                                  0), (0.01,
                                                       0), (-2, 10), (-1.99,
                                                                      10)]),
                    # density=cfg['ray_density'],
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=1e-11,
                ),
                fixtureDef(
                    shape=polygonShape(vertices=[(-0.01,
                                                  0), (0.01, 0), (2,
                                                                  10), (1.99,
                                                                        10)]),
                    # density=cfg['ray_density'],
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=1e-11,
                )
            ],
        )

        self.raycast.color = cfg['raycast_color']
        self.raycast.group = "raycast"
        self.body.group = "agent"
        self.sensor_array.group = "sensor_array"
        for f in self.raycast.fixtures:
            f.sensor = True
        for f in self.body.fixtures:
            f.sensor = False
        for f in self.sensor_array.fixtures:
            f.sensor = False
        ray_joint = weldJointDef(bodyA=self.sensor_array,
                                 bodyB=self.raycast,
                                 localAnchorB=(HEADPOS[0] * self.size,
                                               HEADPOS[1] * self.size),
                                 localAnchorA=(HEADPOS[0] * self.size,
                                               HEADPOS[1] * self.size))
        self.raycast.joint = self.world.CreateJoint(ray_joint)
        self.raycast.userData = self.raycast
        self.body.color = cfg['body_color']
        sensor_array_joint = wheelJointDef(
            bodyA=self.body,
            bodyB=self.sensor_array,
            localAnchorA=(HEADPOS[0] * self.size, HEADPOS[1] * self.size),
            localAnchorB=(0, 0),
            enableMotor=True,
            maxMotorTorque=180 * 900 * self.size * self.size,
            motorSpeed=5,
        )
        self.sensor_array.joint = self.world.CreateJoint(sensor_array_joint)
        self.sensor_array.color = cfg['sensor_array_color']
        self.sensor_array.tiles = set()
        self.sensor_array.userData = self.sensor_array
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
                    # density=cfg['wheel_density'],
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
            rjd = revoluteJointDef(
                bodyA=self.body,
                bodyB=w,
                localAnchorA=(wx * self.size, wy * self.size),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                # maxMotorTorque=180 * 900 * self.size * self.size,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.render_list = self.wheels + [self.body] + [self.sensor_array
                                                        ] + [self.raycast]
        # self.render_list = [self.body] #+ [self.sensor_array]

    def accelerate(self, acc: float) -> None:
        # if (acc <= 0.0):
        #     acc = 0
        for w in self.wheels:
            w.acc = acc

    def camera_spin(self, acc: float) -> None:
        diff = acc - self.sensor_array.angle
        self.sensor_array.angle += diff

    def steer(self, rot: int):
        if rot < 0:
            self.rotate_flag = True
            self.angle -= np.pi / 2
        elif rot > 0:
            self.rotate_flag = True
            self.angle += np.pi / 2

        # for w in self.wheels:
        #     if s < 0:
        #         w.angle = -90
        #     elif s > 0:
        #         w.angle = 90
        #     else:
        #         w.angle = 0

    def step(self, dt) -> None:

        if self.rotate_flag:
            self.rotate_flag = False
            old_angle = self.body.angle
            new_angle = self.angle
            diff_angle = new_angle - old_angle
            print('==============')
            print(f'Old Angle: {old_angle}')
            print(f'New Angle: {new_angle}')
            print(f'Delta Angle: {self.body.angle + diff_angle}')
            # self.body.ApplyTorque(1e20 * diff_angle, False)
            self.body.angle += diff_angle
            self.sensor_array.angle += diff_angle
            # for r in self.raycast:
            #     r.angle += diff_angle
            for w in self.wheels:
                w.angle += diff_angle

        for w in self.wheels:
            # The Freenove car does not have wheels that swivel so we will
            # not bother animating them.
            if w.acc == 0.0:
                w.linearVelocity = vec2(0.0, 0.0)
                self.body.linearVelocity = vec2(0.0, 0.0)
                self.sensor_array.linearVelocity = vec2(0.0, 0.0)
                for r in self.raycast:
                    r.linearVelocity = vec2(0.0, 0.0)
            x_force = w.acc * np.cos(self.angle - 0.5 * np.pi)
            y_force = w.acc * np.sin(self.angle - 0.5 * np.pi)
            if self.angle == 0.0:
                w.ApplyForceToCenter((0.0, y_force), True)
            else:
                w.ApplyForceToCenter((x_force, y_force), True)

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
