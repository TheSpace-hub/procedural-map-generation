import pygame as pg
from pygame import Surface, Rect, Vector2

from enum import Enum

from random import randint, uniform, expovariate
from math import sqrt, pi, cos, sin
from time import sleep

pg.init()


class Log:
    logs: list[str] = []
    stop: bool = False
    my_font = pg.font.SysFont('Comic Sans MS', 30)

    @classmethod
    def log(cls, text: any):
        if cls.stop:
            return
        cls.logs.append(str(text))
        cls.logs = cls.logs[-3:]

    @classmethod
    def draw(cls, screen: pg.Surface):
        offset: int = 0
        size: tuple[int, int] = [0, 0]
        for log in Log.logs:
            text_surface = cls.my_font.render(log, False, (255, 0, 0))
            size[0] = max(size[0], text_surface.get_size()[0] + 20)
            offset += text_surface.get_size()[1]
        size[1] = offset + 20
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



class Map:
    map: list[list[Tile]] = []
    rooms: list[Rect] = []
    rooms_overlap: bool = True

    @staticmethod
    def _choose_room_size_exponential(min_size, max_size, decay_rate=0.5):
        size = min_size + int(expovariate(decay_rate))
        return min(size, max_size)

    @classmethod
    def generate_initial_rooms(cls, count: int = 50):
        radius: int = 15
        for i in range(count):
            angle = uniform(0, 2 * pi)
            distance = sqrt(uniform(0, 1)) * radius

            x = 15 + distance * cos(angle)
            y = 15 + distance * sin(angle)

            cls.rooms.append(
                Rect(x, y, cls._choose_room_size_exponential(3, 10), cls._choose_room_size_exponential(3, 10))
            )
        cls.update_map()

    @classmethod
    def get_size(cls) -> tuple[int, int]:
        max_cord: tuple[int, int] = [0, 0]
        for room in cls.rooms:
            max_cord = [max(max_cord[0], room.x + room.width), max(max_cord[1], room.y + room.height)]

        return max_cord

    @classmethod
    def update_map(cls):
        cls.map: list[list[Tile]] = []

        max_cord: tuple[int, int] = [0, 0]
        for room in cls.rooms:
            max_cord = [max(max_cord[0], room.x + room.width), max(max_cord[1], room.y + room.height)]

        for y in range(max_cord[1]):
            row: list[Tile] = [Tile.EMPTY] * max_cord[0]
            for x in range(max_cord[0]):
                for room in cls.rooms:
                    if room.x <= x <= room.x + room.width - 1 and room.y <= y <= room.y + room.height - 1:
                        row[x] = Tile.FLOOR
                        if room.x == x or room.x + room.width - 1 == x or room.y == y or room.y + room.height - 1 == y:
                            row[x] = Tile.BARRIER
            cls.map.append(row)

    @classmethod
    def separation_steering_for_rooms(cls) -> bool:
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

        cls.update_map()
        cls.rooms_overlap = rooms_overlap

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

                if cls.map[y][x] != Tile.EMPTY:
                    pg.draw.rect(surface, color,
                                 Rect((960 - cls.get_size()[0] * tile_size / 2) + x * tile_size,
                                      (540 - cls.get_size()[1] * tile_size / 2) + y * tile_size, tile_size, tile_size),
                                 tile_size // 3)


def main():
    pg.init()

    Map.generate_initial_rooms()

    screen = pg.display.set_mode((1920, 1080))
    screen.fill((32, 32, 32))

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    Log.stop = not Log.stop

        if Map.rooms_overlap:
            Map.separation_steering_for_rooms()

        screen.fill((32, 32, 32))
        Map.draw(screen)

        Log.draw(screen)
        pg.display.flip()


if __name__ == '__main__':
    main()
