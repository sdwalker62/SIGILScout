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
engine_power = 100000000 * cfg['size']**2


class Scout:

    def __init__(self, world, init_angle: float, init_x: float,
                 init_y: float) -> None:
        self.world = world
        self.size = cfg['size']
        HEAD_POLY = [(-20, 10), (-20, -10), (20, 10), (20, -10)]
        self.sensor_array = self.world.CreateDynamicBody(
            position=(init_x + HEADPOS[0] * self.size,
                      init_y + HEADPOS[1] * self.size),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(vertices=[(x * self.size, y * self.size)
                                                 for x, y in HEAD_POLY]),
                    density=0.0,
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
                                                        for x, y in BODY_POLY]),
                           density=1000.0)
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
                    density=1e-10,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
                fixtureDef(
                    shape=polygonShape(vertices=[(-0.01,
                                                  0), (0.01,
                                                       0), (-2, 10), (-1.99,
                                                                      10)]),
                    density=1e-10,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
                fixtureDef(
                    shape=polygonShape(vertices=[(-0.01,
                                                  0), (0.01, 0), (2,
                                                                  10), (1.99,
                                                                        10)]),
                    density=1e-10,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
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
        self.render_list = self.wheels + [self.body] + [self.sensor_array
                                                       ] + [self.raycast]
        # self.render_list = [self.body] #+ [self.sensor_array]

    def accelerate(self, acc: float) -> None:
        acc = np.clip(acc, 0, 1)
        for w in self.wheels:
            diff = acc - w.acc
            w.acc += diff

    def camera_spin(self, acc: float) -> None:
        self.sensor_array.angle += acc

    def decelerate(self, d: float) -> None:
        for w in self.wheels:
            w.brake = d

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt) -> None:
        for w in self.wheels:
            # The Freenove car does not have wheels that swivel so we will
            # not bother animating them.

            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed
            ENGINE_POWER = 100000000 * self.size * self.size
            WHEEL_MOMENT_OF_INERTIA = 4000 * self.size * self.size
            w.omega += (
                    dt
                    * ENGINE_POWER
                    * w.acc
                    / WHEEL_MOMENT_OF_INERTIA
                    / (abs(w.omega) + 5.0)
            )

            # BRAKING LOGIC
            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                direction = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += direction * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            w.omega -= dt * f_force * w.wheel_rad
            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )

    def lineAndAABB(self):
        # aabb = self.body.GetFixtureList()[0].GetAABB(int32 childIndex) <- const b2AABB&
        # aabb.Contains(const b2AABB &aabb) <- returns a boolean
        # aabb.RayCast(b2RayCastOutput *output, const b2RayCastInput &input) <- returns a boolean

        # self.body.GetFixtureList()[0].GetShape() <- b2Shape*

        #self.body.GetFixtureList()[0].GetShape().RayCast(
        #b2RayCastOutput* output,
        #const b2RayCastInput& input,
        #const b2Transform transform,
        #int32 childIndex,
        #)
        pass
        # print("Sup Raycast!")

        # output = None
        # endPoint = vec2(0, 0) - vec2(0, -500)
        # # input = rayCastInput.__init__.__code__.co_varnames
        # # print(input)
        # input = rayCastInput(p1=vec2(0,0), p2=endPoint, maxFraction=1.0)
        # # print(input)
        # something = self.body.fixtures[0].shape.RayCast(
        #     output,
        #     input,
        #     self.body.transform,
        #     0
        # )

        # print(something)

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
