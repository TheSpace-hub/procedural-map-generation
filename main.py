import pygame as pg
from pygame import Surface, Rect, Vector2

from enum import Enum

from random import uniform
from math import sqrt, pi, cos, sin
from time import sleep
import numpy as np

pg.init()


class Log:
    logs: list[str] = []
    stop: bool = False
    my_font = pg.font.SysFont('freesansbold', 30)

    @classmethod
    def log(cls, text: any):
        if cls.stop:
            return
        cls.logs.append(str(text))
        cls.logs = cls.logs[-3:]

    @classmethod
    def draw(cls, screen: pg.Surface):
        offset: int = 0
        size: tuple[int, int] = (0, 0)
        for log in Log.logs:
            text_surface = cls.my_font.render(log, False, (255, 0, 0))
            size = (max(size[0], text_surface.get_size()[0] + 20), 0)
            offset += text_surface.get_size()[1]
        size = (size[0], offset + 20)
        offset = 0

        pg.draw.rect(screen, (32, 32, 32), Rect(0, 0, size[0], size[1]))
        for log in Log.logs:
            text_surface = cls.my_font.render(log, False, (255, 0, 0))
            screen.blit(text_surface, (10, 10 + offset))
            offset += text_surface.get_size()[1]


class Tile(Enum):
    EMPTY = 0
    BARRIER = 1
    FLOOR = 2
    ROOM_BARRIER = 3
    ROOM_FLOOR = 4


class ConstructionStage(Enum):
    GENERATE_ROOMS = 1
    SEPARATION_STEERING_FOR_ROOMS = 2
    ARRANGEMENT_OF_THE_CORRIDOR_GRAPH_VERTICES = 3
    CREATING_CORRIDORS = 4
    REMOVING_UNNECESSARY_ROOMS = 5
    REMOVE_GRAPH = 6
    SINGLE_COLOR = 7
    DONE = 0


class Map:
    map: list[list[Tile]] = []
    rooms: list[Rect] = []
    big_rooms_center_points: list[tuple[int, int]] = []
    edges_of_the_corridor_graph: list[tuple[int, int]] = []
    construction_stage: ConstructionStage = ConstructionStage.GENERATE_ROOMS

    @staticmethod
    def _is_room_big(room: Rect) -> bool:
        return room.width >= 12 and room.height >= 12

    @staticmethod
    def _choose_room_size(min_size, max_size, mean, std_dev):
        size = int(np.random.normal(mean, std_dev))
        return max(min(size, max_size), min_size)

    @staticmethod
    def _distance(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _get_tiles_in_line(x0, y0, x1, y1, width) -> list[tuple[int, int]]:
        def bresenham_line(x0, y0, x1, y1):
            points = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                points.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            return points

        main_line = bresenham_line(x0, y0, x1, y1)

        wide_line = set()
        for (x, y) in main_line:
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    wide_line.add((x + dx, y + dy))

        return list(wide_line)

    @classmethod
    def _is_near_tile_empty(cls, tile) -> bool:
        return (cls.map[tile[1] - 1][tile[0]] == Tile.EMPTY or
                cls.map[tile[1] - 1][tile[0] - 1] == Tile.EMPTY or
                cls.map[tile[1] - 1][tile[0] + 1] == Tile.EMPTY or
                cls.map[tile[1] + 1][tile[0]] == Tile.EMPTY or
                cls.map[tile[1] + 1][tile[0] - 1] == Tile.EMPTY or
                cls.map[tile[1] + 1][tile[0] + 1] == Tile.EMPTY or
                cls.map[tile[1]][tile[0] + 1] == Tile.EMPTY or
                cls.map[tile[1]][tile[0] - 1] == Tile.EMPTY
                )

    @classmethod
    def _prim_mst(cls) -> list[tuple[int, int]]:
        points = cls.big_rooms_center_points
        if not points:
            return []

        mst = []
        used = set()
        used.add(0)

        while len(used) < len(points):
            min_edge = None
            min_weight = float('inf')

            for u in used:
                for v in range(len(points)):
                    if v not in used:
                        dist = cls._distance(points[u], points[v])
                        if dist < min_weight:
                            min_weight = dist
                            min_edge = (u, v)

            if min_edge:
                mst.append(min_edge)
                used.add(min_edge[1])

        return mst

    @classmethod
    def generate_initial_rooms(cls, count: int = 15):
        radius: int = 15
        angle = uniform(0, 2 * pi)
        distance = sqrt(uniform(0, 1)) * radius

        x = 15 + distance * cos(angle)
        y = 15 + distance * sin(angle)

        room = Rect(x, y, cls._choose_room_size(5, 25, 12, 2), cls._choose_room_size(5, 25, 12, 2))

        cls.rooms.append(room)
        cls.update_rooms_in_map()
        if len(cls.rooms) == count:
            cls.construction_stage = ConstructionStage.SEPARATION_STEERING_FOR_ROOMS

    @classmethod
    def get_size(cls) -> tuple[int, int]:
        max_cord: tuple[int, int] = (0, 0)
        for room in cls.rooms:
            max_cord = (max(max_cord[0], room.x + room.width), max(max_cord[1], room.y + room.height))

        return max_cord

    @classmethod
    def update_rooms_in_map(cls):
        cls.map: list[list[Tile]] = []

        max_cord: tuple[int, int] = (0, 0)
        for room in cls.rooms:
            max_cord = (max(max_cord[0], room.x + room.width), max(max_cord[1], room.y + room.height))

        for y in range(max_cord[1]):
            row: list[Tile] = [Tile.EMPTY] * max_cord[0]
            for x in range(max_cord[0]):
                for room in cls.rooms:
                    if room.x <= x <= room.x + room.width - 1 and room.y <= y <= room.y + room.height - 1:
                        row[x] = Tile.FLOOR
                        if cls._is_room_big(room):
                            row[x] = Tile.ROOM_FLOOR
                        if room.x == x or room.x + room.width - 1 == x or room.y == y or room.y + room.height - 1 == y:
                            row[x] = Tile.BARRIER
                            if cls._is_room_big(room):
                                row[x] = Tile.ROOM_BARRIER
            cls.map.append(row)

    @classmethod
    def build_step(cls) -> bool:
        Log.log(cls.construction_stage)
        actions = {
            ConstructionStage.GENERATE_ROOMS: cls.generate_initial_rooms,
            ConstructionStage.SEPARATION_STEERING_FOR_ROOMS: cls.separation_steering_for_rooms,
            ConstructionStage.ARRANGEMENT_OF_THE_CORRIDOR_GRAPH_VERTICES: cls.arrangement_of_the_corridor_graph_vertices,
            ConstructionStage.CREATING_CORRIDORS: cls.creating_corridors,
            ConstructionStage.REMOVING_UNNECESSARY_ROOMS: cls.removing_unnecessary_rooms,
            ConstructionStage.REMOVE_GRAPH: cls.remove_graph,
            ConstructionStage.SINGLE_COLOR: cls.single_color,
            ConstructionStage.DONE: lambda: None
        }

        actions[cls.construction_stage]()

        return cls.construction_stage == ConstructionStage.REMOVE_GRAPH

    @classmethod
    def remove_graph(cls):
        sleep(1)
        cls.big_rooms_center_points = []
        cls.edges_of_the_corridor_graph = []

        cls.construction_stage = ConstructionStage.SINGLE_COLOR

    @classmethod
    def single_color(cls):
        sleep(1)
        for y in range(len(cls.map)):
            for x in range(len(cls.map[y])):
                if cls.map[y][x] == Tile.ROOM_BARRIER:
                    cls.map[y][x] = Tile.BARRIER
                elif cls.map[y][x] == Tile.ROOM_FLOOR:
                    cls.map[y][x] = Tile.FLOOR
        cls.construction_stage = ConstructionStage.DONE

    @classmethod
    def separation_steering_for_rooms(cls):
        rooms_overlap = False
        separation_vectors: list[Vector2] = []
        for target in cls.rooms:
            separation_vectors.append(Vector2())
            for neighbor in cls.rooms:
                if target == neighbor:
                    continue

                if target.colliderect(neighbor):
                    rooms_overlap = True
                    diff = Vector2(target.center[0], target.center[1]) - Vector2(neighbor.center[0],
                                                                                 neighbor.center[1])

                    if diff.length() != 0.0:
                        diff = diff.normalize()
                    separation_vectors[-1:][0] += diff

        for i in range(len(separation_vectors)):
            if separation_vectors[i].length() != 0:
                separation_vectors[i] = separation_vectors[i].normalize()

        for i in range(len(cls.rooms)):
            room = cls.rooms[i]
            move_vector = separation_vectors[i]
            room.x += round(move_vector.x)
            room.y += round(move_vector.y)

            if room.x < 0:
                room.x = 0
                for target in cls.rooms:
                    target.x += 1

            if room.y < 0:
                room.y = 0
                for target in cls.rooms:
                    target.y += 1

        cls.update_rooms_in_map()

        if not rooms_overlap:
            cls.construction_stage = ConstructionStage.ARRANGEMENT_OF_THE_CORRIDOR_GRAPH_VERTICES

    @classmethod
    def arrangement_of_the_corridor_graph_vertices(cls):
        count = len(cls.big_rooms_center_points)
        for room in cls.rooms:
            if cls._is_room_big(room):
                if count == 0:
                    cls.big_rooms_center_points.append((room.centerx, room.centery))
                    return
                count -= 1
        cls.edges_of_the_corridor_graph = cls._prim_mst()
        cls.construction_stage = ConstructionStage.CREATING_CORRIDORS

    @classmethod
    def creating_corridors(cls):
        for edge in cls.edges_of_the_corridor_graph:
            for tile in cls._get_tiles_in_line(
                    cls.big_rooms_center_points[edge[0]][0], cls.big_rooms_center_points[edge[0]][1],
                    cls.big_rooms_center_points[edge[1]][0], cls.big_rooms_center_points[edge[1]][1], 2):
                cls.map[tile[1]][tile[0]] = Tile.FLOOR
        for edge in cls.edges_of_the_corridor_graph:
            for tile in cls._get_tiles_in_line(
                    cls.big_rooms_center_points[edge[0]][0], cls.big_rooms_center_points[edge[0]][1],
                    cls.big_rooms_center_points[edge[1]][0], cls.big_rooms_center_points[edge[1]][1], 2):
                if cls._is_near_tile_empty(tile):
                    cls.map[tile[1]][tile[0]] = Tile.BARRIER

        cls.construction_stage = ConstructionStage.REMOVING_UNNECESSARY_ROOMS

    @classmethod
    def removing_unnecessary_rooms(cls):
        def erase_room(target: Rect):
            for y in range(target.height):
                for x in range(target.width):
                    cls.map[target.y + y][target.x + x] = Tile.EMPTY

        def is_unaffected(target: Rect):
            for y in range(target.height):
                if cls.map[target.y + y][target.x] == Tile.FLOOR or cls.map[target.y + y][
                    target.x + target.width - 1] == Tile.FLOOR:
                    return False
            for x in range(target.width):
                if cls.map[target.y][target.x + x] == Tile.FLOOR or cls.map[target.y + target.height - 1][
                    target.x + x] == Tile.FLOOR:
                    return False
            erase_room(target)
            return True

        cls.rooms = list(filter(is_unaffected, cls.rooms))
        cls.construction_stage = ConstructionStage.REMOVE_GRAPH

    @classmethod
    def draw(cls, surface: Surface):
        tile_size: int = 5
        for y in range(len(cls.map)):
            for x in range(len(cls.map[y])):
                color = (128, 128, 128)
                if cls.map[y][x] == Tile.BARRIER:
                    color = (0, 0, 255)
                elif cls.map[y][x] == Tile.FLOOR:
                    color = (128, 128, 255)
                elif cls.map[y][x] == Tile.ROOM_FLOOR:
                    color = (255, 128, 128)
                elif cls.map[y][x] == Tile.ROOM_BARRIER:
                    color = (255, 0, 0)

                if cls.map[y][x] != Tile.EMPTY:
                    pg.draw.rect(surface, color,
                                 Rect((960 - cls.get_size()[0] * tile_size / 2) + x * tile_size,
                                      (540 - cls.get_size()[1] * tile_size / 2) + y * tile_size, tile_size, tile_size),
                                 tile_size // 3)

        for point in cls.big_rooms_center_points:
            pg.draw.circle(surface, (0, 255, 0), [
                (960 - cls.get_size()[0] * tile_size / 2) + point[0] * tile_size,
                (540 - cls.get_size()[1] * tile_size / 2) + point[1] * tile_size
            ], 5)

        for edge in cls.edges_of_the_corridor_graph:
            start = cls.big_rooms_center_points[edge[0]]
            end = cls.big_rooms_center_points[edge[1]]
            pg.draw.line(surface, (0, 255, 0),
                         [
                             (960 - cls.get_size()[0] * tile_size / 2) + start[0] * tile_size,
                             (540 - cls.get_size()[1] * tile_size / 2) + start[1] * tile_size
                         ],
                         [
                             (960 - cls.get_size()[0] * tile_size / 2) + end[0] * tile_size,
                             (540 - cls.get_size()[1] * tile_size / 2) + end[1] * tile_size
                         ],
                         2)


def main():
    pg.init()

    screen = pg.display.set_mode((1920, 1080))
    screen.fill((32, 32, 32))
    go = False
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    Log.stop = not Log.stop
                if event.key == pg.K_b:
                    go = True

        if go:
            Map.build_step()

        screen.fill((32, 32, 32))
        Map.draw(screen)

        Log.draw(screen)
        pg.display.flip()


if __name__ == '__main__':
    main()
