# kubi.py – extracted from https://github.com/stefanix/kubi
# Author: Stefan Keim
# License: MIT
# Adapted for integration with cube2sphere project (Yai, 2025)

import math
import numpy as np
import pyvips

class Cubemap:
    def __init__(self, right, left, top, down, front, back):
        self.faces = {
            'right': right,
            'left': left,
            'top': top,
            'down': down,
            'front': front,
            'back': back
        }

    def to_equirectangular(self, width, height, interpolation='bilinear'):
        theta = np.linspace(-math.pi, math.pi, width, endpoint=False)
        phi = np.linspace(math.pi/2, -math.pi/2, height)

        theta, phi = np.meshgrid(theta, phi)
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)

        abs_x = np.abs(x)
        abs_y = np.abs(y)
        abs_z = np.abs(z)

        max_axis = np.maximum(np.maximum(abs_x, abs_y), abs_z)
        x /= max_axis
        y /= max_axis
        z /= max_axis

        face_idx = np.zeros_like(x, dtype='<U5')
        u = np.zeros_like(x)
        v = np.zeros_like(y)

        is_x = (abs_x >= abs_y) & (abs_x >= abs_z)
        is_y = (abs_y >= abs_x) & (abs_y >= abs_z)
        is_z = (abs_z >= abs_x) & (abs_z >= abs_y)

        face_idx[is_x & (x > 0)] = 'right'
        u[is_x & (x > 0)] = -z[is_x & (x > 0)] / abs_x[is_x & (x > 0)]
        v[is_x & (x > 0)] = -y[is_x & (x > 0)] / abs_x[is_x & (x > 0)]

        face_idx[is_x & (x < 0)] = 'left'
        u[is_x & (x < 0)] = z[is_x & (x < 0)] / abs_x[is_x & (x < 0)]
        v[is_x & (x < 0)] = -y[is_x & (x < 0)] / abs_x[is_x & (x < 0)]

        face_idx[is_y & (y > 0)] = 'top'
        u[is_y & (y > 0)] = x[is_y & (y > 0)] / abs_y[is_y & (y > 0)]
        v[is_y & (y > 0)] = z[is_y & (y > 0)] / abs_y[is_y & (y > 0)]

        face_idx[is_y & (y < 0)] = 'down'
        u[is_y & (y < 0)] = x[is_y & (y < 0)] / abs_y[is_y & (y < 0)]
        v[is_y & (y < 0)] = -z[is_y & (y < 0)] / abs_y[is_y & (y < 0)]

        face_idx[is_z & (z > 0)] = 'front'
        u[is_z & (z > 0)] = x[is_z & (z > 0)] / abs_z[is_z & (z > 0)]
        v[is_z & (z > 0)] = -y[is_z & (z > 0)] / abs_z[is_z & (z > 0)]

        face_idx[is_z & (z < 0)] = 'back'
        u[is_z & (z < 0)] = -x[is_z & (z < 0)] / abs_z[is_z & (z < 0)]
        v[is_z & (z < 0)] = -y[is_z & (z < 0)] / abs_z[is_z & (z < 0)]

        u = ((u + 1) / 2 * (self.faces['front'].width - 1)).astype(np.float32)
        v = ((v + 1) / 2 * (self.faces['front'].height - 1)).astype(np.float32)

        out = pyvips.Image.black(width, height).new_from_image([0, 0, 0])

        for face in self.faces:
            mask = (face_idx == face)
            if not np.any(mask):
                continue

            u_map = u.copy()
            v_map = v.copy()
            u_map[~mask] = 0
            v_map[~mask] = 0

            # pyvips needs 1-channel float images for maps
            mapx = pyvips.Image.new_from_memory(u_map.astype(np.float32).tobytes(), width, height, 1, 'float')
            mapy = pyvips.Image.new_from_memory(v_map.astype(np.float32).tobytes(), width, height, 1, 'float')

            warped = self.faces[face].mapim(mapx, mapy, interpolate=pyvips.Interpolate.new(interpolation))
            out = out.ifthenelse(mask.astype(np.uint8) * 255, warped, out)

        return out
