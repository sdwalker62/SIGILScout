from typing import List, Tuple


class SquareObstacle:

    def __init__(self, size: int) -> None:
        self.size = size

    def get_vertices(self, anchor_x: float, anchor_y: float) -> List[Tuple]:
        return [(anchor_x, anchor_y), (anchor_x + self.size, anchor_y),
                (anchor_x + self.size, anchor_y + self.size),
                (anchor_x, anchor_y + self.size)]
