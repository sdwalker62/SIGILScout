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
    "tart_orange": (240, 84, 79),
    "sac_green": (14, 64, 45),
    "emerald": (91, 186, 111)
}


class BoundaryDetector(contactListener):
    def __init__(self, env):
        # print('Boundary Detector initiated')
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # print('begin contact')
        # f_A = contact.fixtureA
        # f_B = contact.fixtureB
        # print(f_A)
        # print(f_B)
        self._contact(contact, True)

    def EndContact(self, contact):
        # print('end contact')
        self._contact(contact, False)

    def _contact(self, contact, begin):
        # print('contact!')
        # tile = None
        # obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u2 is not None:
            if "group" in u2.__dict__.keys() and u2.__dict__["group"] == "border":
                print("Out of bounds!")
        # if u2 is not None:
        #     if "group" in u2.__dict__.keys() and u2.__dict__["group"] == "border":
        #         if u1 is not None:
        #             if "group" in u1.__dict__.keys() and u1.__dict__["group"] != "traversable_area":
        #                 self.env.hit_obstacle = True

        # if u2 is not None:
        #     if "group" in u2.__dict__.keys() and u2.__dict__["group"] == "obstacle":
        #         if u1 is not None:
        #             if "group" in u1.__dict__.keys() and u1.__dict__["group"] != "traversable_area":
        #                 self.env.hit_obstacle = True
        # if u1 and "road_friction" in u1.__dict__:
        #     tile = u1
        #     obj = u2
        # if u2 and "road_friction" in u2.__dict__:
        #     tile = u2
        #     obj = u1
        # if not tile:
        #     return

        # inherit tile color from env
        # tile.color = self.env.road_color / 255
        # if not obj or "tiles" not in obj.__dict__:
        #     return
        # if begin:
        #     obj.tiles.add(tile)
        #     if not tile.road_visited:
        #         tile.road_visited = True
        #         self.env.reward += 1000.0 / len(self.env.track)
        #         self.env.tile_visited_count += 1
        #
        #         # Lap is considered completed if enough % of the track was covered
        #         if (
        #             tile.idx == 0
        #             and self.env.tile_visited_count / len(self.env.track)
        #             > self.lap_complete_percent
        #         ):
        #             self.env.new_lap = True
        # else:
        #     obj.tiles.remove(tile)


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
        continuous: bool = True,
    ) -> None:
        EzPickle.__init__(
            self
        )  # Un-pickles an object and passes args to its constructor
        self.continuous = continuous
        self.boundaryDetector = BoundaryDetector(self)
        self.render_mode = render_mode
        self.surface = None
        self.renderer = Renderer(self.render_mode, self._render)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )
        self.t = 0.0
        self.is_open = True

        self.domain_random = domain_random
        self.world = Box2D.b2World((0, 0), contactListener=self.boundaryDetector)
        self.screen = None
        self.clock = None
        self.agent = None
        self.verbose = verbose

        self.hit_obstacle = False

        self.reward = 0.0
        self.prev_reward = 0.0
        self.new_trial = False
        # do nothing, left, right, brake, gas
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake
        self.state_space = spaces.Box(
            low=0,
            high=25,
            shape=(STATE_H, STATE_W, 3),
            dtype=np.uint8,
        )

        self.tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.agent_color = colors["red"]
        self.objective_color = colors["emerald"]
        self.obstacle_color = colors["black"]
        self.background_color = colors["oxford_blue"]

        # background
        self.arena = list()
        self.arena_poly = list()
        self.traversable_tile_color = colors["opal"]
        self.arena_width = 100
        self.arena_height = 100

        # border
        self.border = list()
        self.border_sensor = list()
        self.border_sensor_poly = list()
        self.border_poly = list()
        self.border_color = colors["tart_orange"]
        self.border_width = 0.1

        # car
        self.car = Car(self.world, 0.0, 0.5 * self.arena_width, 0.5 * self.arena_height)
        
        # obstacles
        self.min_obstacle_count = 10
        self.max_obstacle_count = 15
        self.obstacle_angles = list()
        self.obstacles = list()
        self.obstacles_poly = list()
        self.square_size = 5
        self.base_1 = 3
        self.base_2 = 5

        self.x_anchor_gen = self.halton_sequence(self.base_1)
        self.y_anchor_gen = self.halton_sequence(self.base_2)

        # goal
        self.goal_poly = list()
        self.goal = list()

    def _destroy(self) -> None:
        self.car.destroy()

    def build_course(self) -> None:
        """Builds the arena in which the agent operates."""

        # p1 -- p2
        # |     |
        # p4 -- p3

        # Since the traversable area can be a single texture we can improve performance
        # by drawing one polygon instead of a mesh

        vertices = [
            (0, 0),
            (self.arena_width, 0),
            (self.arena_width, self.arena_height),
            (0, self.arena_height),
        ]
        self.arena_poly.append(vertices)
        self.tile.shape.vertices = vertices
        tile = self.world.CreateStaticBody(fixtures=self.tile)

        tile.userData = tile
        tile.color = self.objective_color
        tile.road_visited = False
        tile.road_friction = 1.0
        # tile.group = "traversable_area"
        tile.idx = 0
        tile.fixtures[0].sensor = True
        self.arena.append(tile)

    def build_border(self) -> None:
        """Builds the arena border."""

        # The points go clockwise p1 -> p2 -> p3 -> p4

        # left border
        left_border_vertices = [
            (-self.border_width, -self.border_width),  # p1
            (0, 0),  # p2
            (0, self.arena_height),  # p3
            (-self.border_width, self.arena_height + self.border_width),  # p4
        ]
        self.border_poly.append(left_border_vertices)

        # top border
        top_border_vertices = [
            (-self.border_width, -self.border_width),  # p1
            (self.arena_width + self.border_width, -self.border_width),  # p2
            (self.arena_width, 0),  # p3
            (0, 0),  # p4
        ]
        self.border_poly.append(top_border_vertices)

        # right border
        right_border_vertices = [
            (self.arena_width, 0),  # p1
            (self.arena_width + self.border_width, -self.border_width),  # p2
            (
                self.arena_width + self.border_width,
                self.arena_height + self.border_width,
            ),  # p3
            (self.arena_width, self.arena_height),  # p4
        ]
        self.border_poly.append(right_border_vertices)

        # bottom border
        bottom_border_vertices = [
            (self.arena_width, self.arena_height),  # p1
            (
                self.arena_width + self.border_width,
                self.arena_height + self.border_width,
            ),  # p2
            (-self.border_width, self.arena_height + self.border_width),  # p3
            (0, self.arena_height),  # p4
        ]
        self.border_poly.append(bottom_border_vertices)

        # for v in self.border_poly:
        #     fixture = fixtureDef(shape=polygonShape(vertices=v))
        #     tile = self.world.CreateStaticBody(fixtures=fixture)
        #     tile.userData = tile
        #     tile.group = "border"
        #     tile.idx = 1
        #     tile.color = self.border_color
        #     tile.fixtures[0].sensor = True
        #     self.border_sensor.append(tile)

        for v in self.border_poly:
            fixture = fixtureDef(shape=polygonShape(vertices=v))
            tile = self.world.CreateStaticBody(fixtures=fixture)
            tile.userData = tile
            tile.group = "border"
            tile.idx = 1
            tile.color = self.border_color
            tile.fixtures[0].sensor = False
            self.border.append(tile)

    def spawn_square(self, anchor_x, anchor_y):
        size = self.square_size
        return [
            (anchor_x, anchor_y),
            (anchor_x + size, anchor_y),
            (anchor_x + size, anchor_y + size),
            (anchor_x, anchor_y + size)
        ]

    def spawn_rectangle(self, anchor_x, anchor_y):
        size = self.square_size
        return [
            (anchor_x, anchor_y),
            (anchor_x + 2 * size, anchor_y),
            (anchor_x + 2 * size, anchor_y + size),
            (anchor_x, anchor_y + size)
        ]

    def spawn_bend(self, anchor_x, anchor_y):
        size = self.square_size
        return [
            (anchor_x, anchor_y),
            (anchor_x + 2 * size, anchor_y),
            (anchor_x + 2 * size, anchor_y + 2 * size),
            (anchor_x + size, anchor_y + 2 * size),
            (anchor_x + size, anchor_y + size),
            (anchor_x, anchor_y + size)
        ]

    @staticmethod
    def halton_sequence(base):
        """Returns a halton sequence for obstacle placement."""
        n, d = 0, 1
        while True:
            x = d - n
            if x == 1:
                n = 1
                d *= base
            else:
                y = d // base
                while x <= y:
                    y //= base
                n = (base + 1) * y - x
            yield n / d

    @staticmethod
    def rotate_obstacle(angle: float, vertices: list) -> List[Tuple]:
        """Rotates the given list of polygon according to the supplied angle.

        We expect an input list of polygons vertices, e.g. for a unit square rotated by 45 degrees:
            vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

        Turn this into a numpy array:
            P = [0 1 1 0
                 0 0 1 1]

        Compute the centroid:
            1 / 4 * [0 + [1 + [1 + [0  = [0.5
                     0]   0]   1]   1]    0.5]

        C = [0.5 0.5 0.5 0.5
             0.5 0.5 0.5 0.5]

        Form a rotation matrix:

        R = [cos(angle) -sin(angle) = [0.7 -0.7
             sin(angle) cos(angle)]    0.7 0.7]

        rotated_poly = R X (P - C) + C = [0.7 -0.7 X [-0.5 0.5 0.5 -0.5 + [0.5 0.5 0.5 0.5
                                          0.7 0.7]    -0.5 -0.5 0.5 0.5]   0.5 0.5 0.5 0.5]

                                       = [0.0 0.7 0.0 -0.7 + [0.5 0.5 0.5 0.5 = [0.5 1.2 0.5 -0.5
                                          -0.7 0.0 0.7 0.0]   0.5 0.5 0.5 0.5]  -0.2 0.5 1.2 0.5]

        We then return the new vertices as a list:
        new_vertices = [(0.5, -0.2), (1.2, 0.5), (0.5, 1.2), (-0.5, 0.5)]
        """
        n_vertices = len(vertices)
        vertex_arr = np.zeros((2, n_vertices))
        for p_idx in range(n_vertices):
            vertex_arr[0, p_idx] = vertices[p_idx][0]
            vertex_arr[1, p_idx] = vertices[p_idx][1]

        centroid = 1 / n_vertices * np.sum(vertex_arr, axis=1)
        c_array = np.array([[centroid[0]*n_vertices], [centroid[1]*n_vertices]])
        r_array = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        new_poly = r_array @ (vertex_arr - c_array) + c_array

        # adjust the vertices to reposition back to the original anchors
        adj_arr = new_poly.T[0] - vertex_arr.T[0]
        new_poly = new_poly - np.expand_dims(adj_arr.T, axis=1)
        rotated_poly = list()
        for col in new_poly.T:
            rotated_poly.append((col[0], col[1]))
        return rotated_poly

    def build_obstacles(self) -> None:
        n_obstacles = np.random.randint(self.min_obstacle_count, self.max_obstacle_count)
        self.obstacle_angles = list(np.random.rand(n_obstacles))
        list_obs = ["square", "rectangle", "bend"]

        for p_idx in range(n_obstacles):
            o = np.random.choice(list_obs)
            x_anchor = next(self.x_anchor_gen) * self.arena_width
            y_anchor = next(self.y_anchor_gen) * self.arena_height
            v = [(0, 0) * 4]
            if o == "square":
                v = self.spawn_square(x_anchor, y_anchor)
            elif o == "rectangle":
                v = self.spawn_rectangle(x_anchor, y_anchor)
            elif o == "bend":
                v = self.spawn_bend(x_anchor, y_anchor)
            v = self.rotate_obstacle(self.obstacle_angles[p_idx], v)
            self.obstacles_poly.append(v)

        for v in self.obstacles_poly:
            self.tile.shape.vertices = v
            tile = self.world.CreateStaticBody(fixtures=self.tile)
            tile.userData = tile
            tile.group = "obstacle"
            tile.color = self.obstacle_color
            tile.fixtures[0].sensor = False
            self.border.append(tile)

    def build_goal(self):
        anchor_x = next(self.x_anchor_gen) * self.arena_width
        anchor_y = next(self.y_anchor_gen) * self.arena_height
        v = [
            (anchor_x, anchor_y),
            (anchor_x + self.square_size * 0.4, anchor_y),
            (anchor_x + self.square_size * 0.4, anchor_y + self.square_size * 0.2),
            (anchor_x, anchor_y + self.square_size * 0.2)
        ]
        self.goal_poly.append(v)
        self.tile.shape.vertices = v
        tile = self.world.CreateStaticBody(fixtures=self.tile)
        tile.userData = tile
        tile.group = "goal"
        tile.color = self.objective_color
        tile.fixtures[0].sensor = False
        self.goal.append(tile)

    def step(self, action: Union[np.ndarray, int]) -> Tuple[ObsType, float, bool, dict]:
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

        step_reward = 0
        done = False
        info = {}

        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                print("THIS IS DISCRETE!")
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        if self.hit_obstacle:
            step_reward = -100
            done = True
            self.hit_obstacle = False

        self.state = self._render("single_state_pixels")

        self.renderer.render_step()
        return self.state, step_reward, done, info

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
            return self._render(mode)

    def _render(self, mode: str = "human"):
        assert mode in self.metadata["render_modes"]
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

        pygame.font.init()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surface = pygame.Surface((WINDOW_W, WINDOW_H))

        angle = -self.car.hull.angle
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        # render background
        self.render_background(zoom, trans, angle)

        # render arena
        self.render_arena(zoom, trans, angle)

        # render car
        self.render_car(zoom, trans, angle, self.render_mode)

        # render border
        self.render_border(zoom, trans, angle)

        # render obstacles
        self.render_obstacles(zoom, trans, angle)

        # render goal
        self.render_goal(zoom, trans, angle)

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

    def create_image_array(self, screen, size) -> np.ndarray:
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def render_background(self, zoom, translation, angle) -> None:
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

    def render_arena(self, zoom, translation, angle) -> None:
        for poly in self.arena_poly:
            self.draw_colored_polygon(
                poly,
                self.traversable_tile_color,
                zoom,
                translation,
                angle,
            )

    def render_border(self, zoom, translation, angle) -> None:
        for poly in self.border_poly:
            self.draw_colored_polygon(
                poly,
                self.border_color,
                zoom,
                translation,
                angle,
            )

    def render_obstacles(self, zoom, translation, angle) -> None:
        for poly in self.obstacles_poly:
            self.draw_colored_polygon(
                poly,
                self.obstacle_color,
                zoom,
                translation,
                angle,
            )

    def render_goal(self, zoom, translation, angle) -> None:
        for poly in self.goal_poly:
            self.draw_colored_polygon(
                poly,
                self.objective_color,
                zoom,
                translation,
                angle,
            )

    def render_car(self, zoom, translation, angle, mode) -> None:
        self.car.draw(
            self.surface,
            zoom,
            translation,
            angle,
            mode not in ["state_pixels", "single_state_pixels"],
        )

    def draw_colored_polygon(
        self,
        poly: list,
        poly_color: Union[
            Color,
            Color,
            Tuple[int, int, int],
            List[int],
            int,
            Tuple[int, int, int, int],
        ],
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
        *,
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
        self._destroy()
        self.obstacles = list()
        self.obstacles_poly = list()
        self.car = Car(self.world, 0.0, 0.5 * self.arena_width, 0.5 * self.arena_height)

        self.build_course()
        self.build_border()
        self.build_obstacles()
        self.build_goal()

        self.renderer.reset()

        self.hit_obstacle = False


    def close(self) -> None:
        """Perform any necessary cleanup. Environments will automatically :meth:`close()`
        themselves when garbage collected or when the program exits.
        """
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])  # steer, accelerate, brake
    import pygame

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    # print('KEY DOWN LEFT')
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    # print("KEY DOWN RIGHT")
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    # print("KEY DOWN UP")
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    # print('KEY DOWN DOWN')
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    # print("KEY DOWN RETURN")
                    global restart
                    restart = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    # print("KEY UP LEFT")
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    # print("KEY UP RIGHT")
                    a[0] = 0
                if event.key == pygame.K_UP:
                    # print("KEY UP UP")
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    # print("KEY UP DOWN")
                    a[2] = 0

    env = SimulateSIGILExplorer()
    env.render()

    is_open = True
    while is_open:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, done, info = env.step(a)
            # s = env.step(a)
            total_reward += r
            if steps % 200 == 0:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            is_open = env.render()
            if done or restart or is_open is False:
                break
    env.close()
