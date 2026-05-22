import numpy as np


class MapBuilder:
    ROAD = 0
    COMPOUND = 1
    RESTAURANT = 2

    WIDTH = 20
    HEIGHT = 16
    COMPOUND_SIZE = 2

    def __init__(self):
        self.restaurant = (0, 0)
        self.compounds = {}
        self.base_map = self._build_map()

    def _build_map(self):
        """
        Build fixed city layout.
        """
        grid = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int8)

        # Restaurant
        grid[self.restaurant] = self.RESTAURANT

        # Predefined compound top-left coordinates
        positions = [
            (1, 3), (1, 7), (1, 11), (1, 15),
            (4, 1), (4, 5), (4, 9), (4, 13), (4, 17),
            (7, 3), (7, 7), (7, 11), (7, 15),
            (10, 1), (10, 5), (10, 9), (10, 13),
            (13, 4), (13, 10)
        ]

        # Mapping compounds
        for idx, (r, c) in enumerate(positions):
            cells = []

            for dr in range(self.COMPOUND_SIZE):
                for dc in range(self.COMPOUND_SIZE):
                    rr = r + dr
                    cc = c + dc

                    if rr < self.HEIGHT and cc < self.WIDTH:
                        grid[rr, cc] = self.COMPOUND
                        cells.append((rr, cc))

            # Entrance = road directly above if available
            entrance = (r - 1, c) if r > 0 else (r + 2, c)

            self.compounds[idx] = {
                "cells": cells,
                "entrance": entrance
            }

        return grid

    def get_base_map(self):
        return self.base_map.copy()

    def get_restaurant(self):
        return self.restaurant

    def get_compounds(self):
        return self.compounds

    def render_ascii(self):
        """
        Debug visualization
        """
        symbols = {
            self.ROAD: ".",
            self.COMPOUND: "C",
            self.RESTAURANT: "R"
        }

        for row in self.base_map:
            print(" ".join(symbols[cell] for cell in row))