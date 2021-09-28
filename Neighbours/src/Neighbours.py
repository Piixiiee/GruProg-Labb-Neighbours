from typing import List
from enum import Enum, auto
from random import *

import pygame as pg
import random


#  Program to simulate segregation.
#  See : http:#nifty.stanford.edu/2014/mccown-schelling-model-segregation/
#

# Enumeration type for the Actors
class Actor(Enum):
    BLUE = auto()
    RED = auto()
    NONE = auto()  # NONE used for empty locations


# Enumeration type for the state of an Actor
class State(Enum):
    UNSATISFIED = auto()
    SATISFIED = auto()
    NA = auto()  # Not applicable (NA), used for NONEs


World = List[List[Actor]]  # Type alias

SIZE = 300


def neighbours():
    pg.init()
    model = NeighborsModel(SIZE)
    _view = NeighboursView(model)
    model.run()


class NeighborsModel:
    # Tune these numbers to test different distributions or update speeds
    FRAME_RATE = 20
    DIST = [0.25, 0.25, 0.5]  # % of RED, BLUE, and NONE
    THRESHOLD = 0.7  # % of surrounding neighbours that should be like me for satisfaction

    # ########### These following two methods are what you're supposed to implement  ###########
    # In this method you should generate a new world
    # using randomization according to the given arguments.

    # TODO Bryta ner metoderna i minder delar (update world)

    def __create_world(self, size) -> World:
        brave_new_world = []
        temp_list = []

        # creates a list containing actors according to the distribution and shuffles it
        for i in range(size * size):
            if i < red:
                temp_list.append(Actor.RED)
            elif red <= i < blue:
                temp_list.append(Actor.BLUE)
            else:
                temp_list.append(Actor.NONE)

        random.shuffle(temp_list)

        # converts the list of actors to a matrix
        for i in range(size):
            brave_new_world.append([])
            for j in range(size):
                brave_new_world[i].append(distribution[j + i * size])
        return brave_new_world

    @staticmethod
    def create_distribution(size):
        red = round(NeighborsModel.DIST[0] * size * size)
        blue = round(NeighborsModel.DIST[1] * size * size) + red

        temp_list = []

        # creates a list containing actors according to the distribution and shuffles it
        for i in range(size * size):
            if i < red:
                temp_list.append(Actor.RED)
            elif red <= i < blue:
                temp_list.append(Actor.BLUE)
            else:
                temp_list.append(Actor.NONE)

        random.shuffle(temp_list)

        return temp_list

    # This is the method called by the timer to update the world
    # (i.e move unsatisfied) each "frame".
    def __update_world(self, size):
        neighbour_list = self.create_satisfaction_list(size, self.world)
        empty_list = self.create_list_of_empties()

        for row in range(len(self.world)):
            for col in range(len(self.world)):
                if neighbour_list[row][col] == State.UNSATISFIED:
                    rand_empty = random.randint(0, len(empty_list) - 1)
                    empty_square = empty_list[rand_empty]
                    mem = self.world[row][col]

                    self.world[empty_square[0]][empty_square[1]] = mem
                    self.world[row][col] = Actor.NONE

                    # Removes the used empty square and replaces it with the newly
                    # created one
                    del empty_list[rand_empty]
                    empty_list.append([row, col])

    def create_list_of_empties(self):
        index_list = []

        for row in range(len(self.world)):
            for col in range(len(self.world)):
                if self.world[row][col] == Actor.NONE:
                    index_list.append([row, col])

        return index_list

    def create_satisfaction_list(self, size, matrix):
        neighbour_list = []

        for row in range(size):
            neighbour_list.append([])
            for col in range(size):
                neighbour_list[row].append(self.find_satisfaction(matrix, row, col))

        return neighbour_list

    @staticmethod
    def find_satisfaction(matrix, row, col):
        neigh_count = 0
        live_neigh = 0

        if matrix[row][col] != Actor.NONE:
            for i in range(row - 1, row + 2, 1):   # Loops through the 3 rows adjacent to the current Actor
                try:
                    if i >= 0:
                        if col > 0:
                            if matrix[i][col - 1] == matrix[row][col]:
                                neigh_count += 1
                            if matrix[i][col - 1] != Actor.NONE:
                                live_neigh += 1
                        if col < len(matrix) - 1:
                            if matrix[i][col + 1] == matrix[row][col]:
                                neigh_count += 1
                            if matrix[i][col + 1] != Actor.NONE:
                                live_neigh += 1
                        if i != row and matrix[i][col] == matrix[row][col]:
                            neigh_count += 1
                        if i != row and matrix[i][col] != Actor.NONE:
                            live_neigh += 1

                except IndexError:
                    pass

            try:
                satisfaction = neigh_count / live_neigh

                if satisfaction >= NeighborsModel.THRESHOLD:
                    result = State.SATISFIED
                else:
                    result = State.UNSATISFIED

            except ZeroDivisionError:
                result = State.UNSATISFIED

        else:
            result = State.NA

        return result

    # ########### the rest of this class is already defined, to handle the simulation clock  ###########
    def __init__(self, size):
        self.size = size
        self.world: World = self.__create_world(size)
        self.observers = []  # for enabling discoupled updating of the view, ignore

    def run(self):
        clock = pg.time.Clock()
        running = True
        while running:
            running = self.__on_clock_tick(clock)
        # stop running
        print("Goodbye!")
        pg.quit()

    def __on_clock_tick(self, clock):
        clock.tick(self.FRAME_RATE)  # update no faster than FRAME_RATE times per second
        self.__update_and_notify()
        return self.__check_for_exit()

    # What to do each frame
    def __update_and_notify(self):
        self.__update_world(self.size)
        self.__notify_all()

    @staticmethod
    def __check_for_exit() -> bool:
        keep_going = True
        for event in pg.event.get():
            # Did the user click the window close button?
            if event.type == pg.QUIT:
                keep_going = False
        return keep_going

    # Use an Observer pattern for views
    def add_observer(self, observer):
        self.observers.append(observer)

    def __notify_all(self):
        for observer in self.observers:
            observer.on_world_update()


# ---------------- Helper methods ---------------------

# Check if inside world
def is_valid_location(size: int, row: int, col: int):
    return 0 <= row < size and 0 <= col < size


# ------- Testing -------------------------------------

# Here you run your tests i.e. call your logic methods
# to see that they really work
def test():
    # A small hard coded world for testing
    test_world = [
        [Actor.RED, Actor.RED, Actor.NONE],
        [Actor.NONE, Actor.BLUE, Actor.NONE],
        [Actor.RED, Actor.NONE, Actor.BLUE]
    ]

    N = NeighborsModel(SIZE)
    N.world = test_world
    # For tests to work THRESHOLD needs to be set to 0.5
    print(N.find_satisfaction(test_world, 0, 0) == State.SATISFIED)
    print(N.find_satisfaction(test_world, 1, 1) == State.UNSATISFIED)
    print(N.find_satisfaction(test_world, 1, 0) == State.NA)

    print(N.create_list_of_empties() == [[0, 2], [1, 0], [1, 2], [2, 1]])

    print(N.create_satisfaction_list(len(test_world), test_world) == [[State.SATISFIED, State.SATISFIED, State.NA],
                                                                      [State.NA, State.UNSATISFIED, State.NA],
                                                                      [State.UNSATISFIED, State.NA, State.SATISFIED]])

    # size = len(test_world)
    # print(is_valid_location(size, 0, 0))
    # print(not is_valid_location(size, -1, 0))
    # print(not is_valid_location(size, 0, 3))
    # print(is_valid_location(size, 2, 2))

    # exit(0)


# Helper method for testing
def count(a_list, to_find):
    the_count = 0
    for a in a_list:
        if a == to_find:
            the_count += 1
    return the_count


# ###########  NOTHING to do below this row, it's pygame display stuff  ###########
# ... but by all means have a look at it, it's fun!
class NeighboursView:
    # static class variables
    WIDTH = 700  # Size for window
    HEIGHT = 700
    MARGIN = 50

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    # Instance methods

    def __init__(self, model: NeighborsModel):
        pg.init()  # initialize pygame, in case not already done
        self.dot_size = self.__calculate_dot_size(len(model.world))
        self.screen = pg.display.set_mode([self.WIDTH, self.HEIGHT])
        self.model = model
        self.model.add_observer(self)

    def render_world(self):
        # # Render the state of the world to the screen
        self.__draw_background()
        self.__draw_all_actors()
        self.__update_screen()

    # Needed for observer pattern
    # What do we do every time we're told the model had been updated?
    def on_world_update(self):
        self.render_world()

    # private helper methods
    def __calculate_dot_size(self, size):
        return max((self.WIDTH - 2 * self.MARGIN) / size, 2)

    @staticmethod
    def __update_screen():
        pg.display.flip()

    def __draw_background(self):
        self.screen.fill(NeighboursView.WHITE)

    def __draw_all_actors(self):
        for row in range(len(self.model.world)):
            for col in range(len(self.model.world[row])):
                self.__draw_actor_at(col, row)

    def __draw_actor_at(self, col, row):
        color = self.__get_color(self.model.world[row][col])
        xy = self.__calculate_coordinates(col, row)
        pg.draw.circle(self.screen, color, xy, self.dot_size / 2)

    # This method showcases how to nicely emulate 'switch'-statements in python
    @staticmethod
    def __get_color(actor):
        return {
            Actor.RED: NeighboursView.RED,
            Actor.BLUE: NeighboursView.BLUE
        }.get(actor, NeighboursView.WHITE)

    def __calculate_coordinates(self, col, row):
        x = self.__calculate_coordinate(col)
        y = self.__calculate_coordinate(row)
        return x, y

    def __calculate_coordinate(self, offset):
        x: float = self.dot_size * offset + self.MARGIN
        return x


if __name__ == "__main__":
    neighbours()
    # test()
