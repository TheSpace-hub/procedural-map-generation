import pygame as pg
from pygame import Surface, Rect, Vector2

from enum import Enum

from random import randint, uniform
from math import sqrt, pi, cos, sin

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

    @classmethod
    def generate_initial_rooms(cls, count: int = 50):
        radius: int = 15
        for i in range(count):
            angle = uniform(0, 2 * pi)
            distance = sqrt(uniform(0, 1)) * radius

            Log.log(f'{distance} || {distance * cos(angle)} / {distance * sin(angle)}')

            x = 15 + distance * cos(angle)
            y = 15 + distance * sin(angle)

            cls.rooms.append(
                Rect(x, y, randint(3, 10), randint(3, 10))
            )

        cls.update_map()

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
    def draw(cls, surface: Surface):
        tile_size: int = 10
        for y in range(len(cls.map)):
            for x in range(len(cls.map[y])):
                color = (128, 128, 128)
                if cls.map[y][x] == Tile.BARRIER:
                    color = (0, 0, 255)
                elif cls.map[y][x] == Tile.FLOOR:
                    color = (128, 128, 255)

                pg.draw.rect(surface, color,
                             Rect(300 + x * tile_size, 300 + y * tile_size, tile_size, tile_size), tile_size // 3)


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
        screen.fill((32, 32, 32))
        Map.draw(screen)

        Log.draw(screen)
        pg.display.flip()


if __name__ == '__main__':
    main()
