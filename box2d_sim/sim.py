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
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[box2d]`"
    )

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")

from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
Color = TypeVar("Color")

STATE_W, STATE_H = 96, 96  # less than Atari 160x192
VIDEO_W, VIDEO_H = 600, 400
WINDOW_W, WINDOW_H = 1000, 800


SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAY_FIELD = 2000 / SCALE  # Game over boundary
FPS = 60  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAY_FIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)

colors = {
    # (R, G, B)
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "oxford_blue": (0, 19, 61),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (40, 40, 40),
    "opal": (198, 216, 211),
    "tart_orange": (240, 84, 79)
}


class BoundaryDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def begin_contact(self, contact):
        pass

    def end_contact(self, contact):
        pass

    def contact(self, contact, begin):
        pass


class SimulateSIGILExplorer(gym.Env, EzPickle):
    r"""Creates a simulation of the SIGIL Explorer environment for learning purposes.

    Inherits from the gym.Env class which implements the following methods and attributes:

        Methods:
            - :meth:`step` - Takes a step in the environment using an action returning the next observation, reward,
              if the environment terminated and more information.
            - :meth:`reset` - Resets the environment to an initial state, returning the initial observation.
            - :meth:`render` - Renders the environment observation with modes depending on the output
            - :meth:`close` - Closes the environment, important for rendering where pygame is imported
            - :meth:`seed` - Seeds the environment's random number generator, :deprecated: in favor of
              `Env.reset(seed=seed)`.

        Attributes:
            - :attr:`action_space` - The Space object corresponding to valid actions
            - :attr:`observation_space` - The Space object corresponding to valid observations
            - :attr:`reward_range` - A tuple corresponding to the minimum and maximum possible rewards
            - :attr:`spec` - An environment spec that contains the information used to initialise the environment from
              `gym.make`
            - :attr:`metadata` - The metadata of the environment, i.e. render modes
            - :attr:`np_random` - The random number generator for the environment

    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
            "single_rgb_array",
            "single_state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        domain_random: bool = False,
    ) -> None:
        EzPickle.__init__(
            self
        )  # Un-pickles an object and passes args to its constructor
        self.boundaryDetector = BoundaryDetector(self)
        self.render_mode = render_mode
        self.surface = None
        self.renderer = Renderer(self.render_mode, self.render)
        self.t = 0.0
        self.is_open = True

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

        self.agent_color = colors["red"]
        self.objective_color = colors["green"]
        self.obstacle_color = colors["red"]
        self.background_color = colors["oxford_blue"]

        # background
        self.arena = list()
        self.arena_poly = list()
        self.traversable_tile_color = colors["opal"]
        self.n_arena_rows = 10
        self.n_arena_cols = 10
        self.bg_tile_width = 1.0
        self.bg_tile_height = 1.0
        self.arena_width = self.n_arena_cols * self.bg_tile_width
        self.arena_height = self.n_arena_rows * self.bg_tile_height

        # border
        self.border = list()
        self.border_poly = list()
        self.border_color = colors["tart_orange"]
        self.border_width = 0.1


    def destroy(self) -> None:
        pass

    def build_course(self) -> None:
        """Builds the arena in which the agent operates."""
        n_tiles = self.n_arena_cols * self.n_arena_rows
        for i in range(n_tiles):
            # p1 -- p2
            # |     |
            # p4 -- p3

            row = math.floor(i / self.n_arena_rows)
            col = i % self.n_arena_cols

            p1 = (row * self.bg_tile_height, col * self.bg_tile_width)
            p2 = (p1[0] + self.bg_tile_width, p1[1])
            p3 = (p1[0] + self.bg_tile_width, p1[1] + self.bg_tile_height)
            p4 = (p1[0], p1[1] + self.bg_tile_height)
            vertices = [p1, p2, p3, p4]
            self.arena_poly.append(vertices)

            fixture = fixtureDef(shape=polygonShape(vertices=vertices))
            tile = self.world.CreateStaticBody(fixtures=fixture)

            tile.userData = tile
            tile.color = self.objective_color
            tile.road_visited = False
            tile.road_friction = 1.0
            tile.idx = i
            tile.fixtures[0].sensor = True
            self.arena.append(tile)

    def build_border(self) -> None:
        """Builds the arena border."""

        # The points go clockwise p1 -> p2 -> p3 -> p4

        # left border
        left_border_vertices = [
            (-self.border_width, -self.border_width),                    # p1
            (0, 0),                                                      # p2
            (0, self.arena_height),                                      # p3
            (-self.border_width, self.arena_height + self.border_width)  # p4
        ]
        self.border_poly.append(left_border_vertices)

        # left border
        top_border_vertices = [
            (-self.border_width, -self.border_width),                    # p1
            (self.arena_width + self.border_width, -self.border_width),  # p2
            (self.arena_width, 0),                                       # p3
            (0, 0)                                                       # p4
        ]
        self.border_poly.append(top_border_vertices)

        # right border
        right_border_vertices = [
            (self.arena_width, 0),                                                          # p1
            (self.arena_width + self.border_width, -self.border_width),                     # p2
            (self.arena_width + self.border_width, self.arena_height + self.border_width),  # p3
            (self.arena_width, self.arena_height)                                                       # p4
        ]
        self.border_poly.append(right_border_vertices)

        # bottom border
        bottom_border_vertices = [
            (self.arena_width, self.arena_height),                                          # p1
            (self.arena_width + self.border_width, self.arena_height + self.border_width),  # p2
            (-self.border_width, self.arena_height + self.border_width),                    # p3
            (0, self.arena_height)                                                          # p4
        ]
        self.border_poly.append(bottom_border_vertices)

        for v in self.border_poly:
            fixture = fixtureDef(shape=polygonShape(vertices=v))
            tile = self.world.CreateStaticBody(fixtures=fixture)
            tile.userData = tile
            tile.color = self.border_color
            tile.fixtures[0].sensor = True
            self.border.append(tile)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        r"""Run one timestep of the environment's dynamics.

        Args:
            action: the action provided by the agent.

        Returns:
            observation: information returned from the environment collected by
                the agents percepts.
            reward: the value gained/lost from executing the supplied action.
            done: a flag to indicate if the episode has terminated.
            info: a dictionary containing additional information about the episode. For
                instance this might include diagnostic information, metrics, and variables
                not collectable via the agent's percepts. Typically, information contained here
                does not influence the evaluation of the agent's performance.

        """
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

    def render(
        self, mode: str = "human"
    ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        r"""Compute the render frames as specified by render_mode attribute during initialization of the environment.

        Canonical render modes are listed below:

            - None (default): no render is computed.
            - human: the environment is continuously rendered in the current display or terminal for human
                consumption.
            - single_rgb_array: return a single frame representing the current state of the environment. A
                frame is a single numpy.ndarray with shape (x, y, 3) representing the RGB values for a pixel image.
            - rgb_array: return a list of frames representing the states of the environment since the last reset.
                Each numpy array is the same as in single_rbg_array mode.
            - ansi: return a list of strings containing a terminal-style text representation for each time step.
                The text can include newlines and ANSI escape sequences (e.g. for colors).

        Notes:

            - Rendering computations is performed internally even if you don't call render().
                To avoid this, you can set render_mode = None and, if the environment supports it,
                call render() specifying the argument 'mode'.
            - Make sure that your class's metadata 'render_modes' key includes
                the list of supported modes. It's recommended to call super()
                in implementations to use the functionality of this method.

        Args:
            mode: one of the modes listed above.

        Returns:


        """
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            if self.screen is None and mode == "human":
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

            pygame.font.init()

            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.surface = pygame.Surface((WINDOW_W, WINDOW_H))

            angle = 0.0
            zoom = 75.0
            scroll_x = -WINDOW_W / 2 + 120
            scroll_y = -WINDOW_H / 4 + 20
            trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
            trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

            # builders
            self.build_course()
            self.build_border()

            # render background
            self.render_background(zoom, trans, angle)

            # render arena
            self.render_arena(zoom, trans, angle)

            # render border
            self.render_border(zoom, trans, angle)

            # render obstacles
            self.render_obstacles(zoom, trans, angle)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            self.screen.fill(0)
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

        if mode in {"rgb_array", "single_rgb_array"}:
            return self.create_image_array(self.surface, (VIDEO_W, VIDEO_H))
        elif mode in {"state_pixels", "single_state_pixels"}:
            return self.create_image_array(self.surface, (STATE_W, STATE_H))
        else:
            return self.is_open

    def create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def render_background(self, zoom, translation, angle):
        bounds = PLAY_FIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]
        # draw background
        self.draw_colored_polygon(
            field,
            self.background_color,
            zoom,
            translation,
            angle,
            clip=False,
        )

    def render_arena(self, zoom, translation, angle):
        for poly in self.arena_poly:
            self.draw_colored_polygon(
                poly,
                self.traversable_tile_color,
                zoom,
                translation,
                angle,
            )

    def render_border(self, zoom, translation, angle):
        for poly in self.border_poly:
            self.draw_colored_polygon(
                poly,
                self.border_color,
                zoom,
                translation,
                angle,
            )

    def render_obstacles(self, zoom, translation, angle):

        poly = [
            (4, 2),
            (5, 2),
            (4, 3),
            (5, 3)
        ]
        self.draw_colored_polygon(
            poly,
            self.obstacle_color,
            zoom,
            translation,
            angle
        )

    def draw_colored_polygon(
        self,
        poly: list,
        poly_color: Union[Color, Color, Tuple[int, int, int], List[int], int, Tuple[int, int, int, int]],
        zoom: float,
        translation: pygame.math.Vector2,
        angle: float,
        clip: bool = True,
    ) -> None:
        """Creates a colored polygon"""
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surface, poly, poly_color)
            gfxdraw.filled_polygon(self.surface, poly, poly_color)

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        r"""Resets the environment to an initial state and returns the initial observation.

        Args:
            seed: The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be
                reset. If you pass an integer, the PRNG will be reset even if it already exists. Usually, you want to
                pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            return_info: Analogous to info in :meth:`step`. Returns additional information.
            options: Additional information used to reset the environment.

        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (optional dictionary): This will *only* be returned if ``return_info=True`` is passed.
                It contains auxiliary information complementing ``observation``. This dictionary should be analogous to
                the ``info`` returned by :meth:`step`.

        """
        self.arena = list()
        self.arena_poly = list()

    def close(self) -> None:
        """Perform any necessary cleanup. Environments will automatically :meth:`close()`
        themselves when garbage collected or when the program exits.
        """
        pass


if __name__ == "__main__":
    a = np.array([0.0] * 3)  # steer, accelerate, brake
    import pygame

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

    env = SimulateSIGILExplorer()
    env.render()

    is_open = True
    while is_open:
        # env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            # s, r, done, info = env.step(a)
            env.step(a)
            # total_reward += r
            if steps % 200 == 0:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            is_open = env.render()
            if restart or is_open is False:
                break
    env.close()
