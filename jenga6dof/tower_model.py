"""Nominal Jenga tower geometry: (layer n, index m) -> 4x4 block pose in the tower frame.

Conventions are documented in config/tower.yaml.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "tower.yaml"


@dataclass
class TowerModel:
    length: float  # meters
    width: float
    height: float
    num_layers: int
    blocks_per_layer: int = 3

    @classmethod
    def from_yaml(cls, path=DEFAULT_CONFIG):
        cfg = yaml.safe_load(Path(path).read_text())
        b, t = cfg["block"], cfg["tower"]
        return cls(b["length_mm"] / 1000, b["width_mm"] / 1000, b["height_mm"] / 1000,
                   t["num_layers"], t["blocks_per_layer"])

    def check_index(self, n, m):
        if not (1 <= n <= self.num_layers and 1 <= m <= self.blocks_per_layer):
            raise ValueError(f"(n={n}, m={m}) out of range: n in 1..{self.num_layers}, "
                             f"m in 1..{self.blocks_per_layer}")

    def nominal_pose(self, n, m):
        """4x4 pose of block (n, m) in the tower frame."""
        self.check_index(n, m)
        offset = (m - (self.blocks_per_layer + 1) / 2) * self.width
        z = (n - 1) * self.height + self.height / 2
        T = np.eye(4)
        if n % 2 == 1:
            # long axis along tower x; stacked front (-y) to back (+y)
            T[:3, 3] = [0.0, offset, z]
        else:
            # long axis along tower y (rotate +90 deg about z); stacked left (-x) to right (+x)
            T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
            T[:3, 3] = [offset, 0.0, z]
        return T

    def block_corners(self):
        """8x3 corners of a block in its local frame (for drawing / cropping)."""
        hx, hy, hz = self.length / 2, self.width / 2, self.height / 2
        return np.array([[sx * hx, sy * hy, sz * hz]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    def all_nominal_poses(self):
        return {(n, m): self.nominal_pose(n, m)
                for n in range(1, self.num_layers + 1)
                for m in range(1, self.blocks_per_layer + 1)}


if __name__ == "__main__":
    tm = TowerModel.from_yaml()
    for key in [(1, 1), (1, 3), (2, 1), (2, 3), (17, 2)]:
        T = tm.nominal_pose(*key)
        print(key, "pos[mm] =", np.round(T[:3, 3] * 1000, 1), "long axis =", T[:3, 0])
