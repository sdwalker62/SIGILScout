import math
from tabnanny import verbose
from turtle import color
from typing import Optional, Union

import numpy as np

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
from gym.utils.renderer import Renderer

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)

colors = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0, 0, 0]),
    "grey": np.array([40, 40, 40]),
}


def boundary_detector():
    pass


class SimulateSIGILExplorer(gym.Envm, EzPickle):
    """
    Creates a simulation of the SIGIL Explorer environment for learning purposes.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        domain_random: bool = False,
    ) -> None:
        EzPickle.__init__(
            self
        )  # Un-pickles an object and passes args to its constructor
        self.boundaryDetector = boundary_detector()
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.surface = None

        self.domain_random = domain_random
        self.world = Box2D.b2World((0, 0), contactListener=self.boundaryDetector)
        self.screen = None
        self.clock = None
        self.agent = None
        self.verbose = verbose

        self.reward = 0.0
        self.prev_reward = 0.0
        self.new_trial = False
        # do nothing, left, right, brake, gas
        self.action_space = spaces.Discrete(5)
        self.state_space = spaces.Box(
            low=0,
            high=25,
            shape=(STATE_H, STATE_W, 3),
            dtype=np.uint8,
        )

        self.traversable_tile_color = None
        self.agent_color = None
        self.objective_color = None
        self.obstacle_color = None

    def destroy(self) -> None:
        pass

    def init_colors(self) -> None:
        self.traversable_tile_color = colors["grey"]
        self.agent_color = colors["red"]
        self.objective_color = colors["green"]
        self.obstacle_color = colors["black"]

    def build_course(self) -> bool:
        pass

    def render(self) -> None:
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )

        pygame.font.init()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surface = pygame.Surface((WINDOW_W, WINDOW_H))
