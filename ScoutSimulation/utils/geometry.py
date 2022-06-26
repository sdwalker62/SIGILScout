import numpy as np
from typing import List, Tuple


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
    c_array = np.array([[centroid[0] * n_vertices], [centroid[1] * n_vertices]])
    r_array = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    new_poly = r_array @ (vertex_arr - c_array) + c_array

    # adjust the vertices to reposition back to the original anchors
    adj_arr = new_poly.T[0] - vertex_arr.T[0]
    new_poly = new_poly - np.expand_dims(adj_arr.T, axis=1)
    rotated_poly = list()
    for col in new_poly.T:
        rotated_poly.append((col[0], col[1]))
    return rotated_poly
