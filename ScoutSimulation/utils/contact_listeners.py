from gym.error import DependencyNotInstalled

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled(
        "box2D is not installed, run `pip install gym[box2d]`")


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
        if u2 is not None and "group" in u2.__dict__.keys(
        ) and u2.__dict__["group"] == "border":
            print("Out of bounds!")
        if u2 is not None and u1 is not None and "group" in u2.__dict__.keys(
        ) and "group" in u1.__dict__.keys():
            if (u1.__dict__["group"] == 'raycast' and
                    u2.__dict__["group"] != 'agent' and
                    u2.__dict__["group"] != 'raycast'):
                print('hit!')
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
