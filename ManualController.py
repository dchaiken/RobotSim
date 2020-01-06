"""
Class for manual control of the agent
Based on a turtlebot, user has ability to increase and decrease angular and linear velocity
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep

from Controller import Controller

event_time = time()
velocity = 0


class ManualController(Controller):

    def display(self):
        """
        Overwrite base display method to create figure event handler
        """
        self.fig, self.axes = plt.subplots(1, 2)
        print("Press arrow keys to change velocity, z to reset velocity to 0, q to close window")
        self.fig.canvas.mpl_connect('key_press_event', self.control)
        self._draw()
        plt.show()

    def control(self, event):
        """
        Control used as callback to change velocity on key press
        """
        if event.key == 'q':
            plt.close()
        elif event.key == 'z':
            self.velocity = [0, 0]
        else:
            # Handle velocity change on arrow keys
            desired_velocity = self.velocity[:]
            if event.key == 'left':
                desired_velocity[1] -= 0.1
            elif event.key == 'right':
                desired_velocity[1] += 0.1
            elif event.key == 'up':
                desired_velocity[0] += 0.1
            elif event.key == 'down':
                desired_velocity[0] -= 0.1
            self.velocity = [self._filter_velocity(desired_velocity[0], self.velocity[0]), self._filter_velocity(
                desired_velocity[1], self.velocity[1])]
            self.fig.clear()
            self._draw()
            self.move()


def test():
    world = np.load('maps/basic_map.npy')
    known = -np.ones(world.shape)
    ctrlr = ManualController(world, known, [500, 500, 0])
    ctrlr.explore()
    ctrlr.display()


if __name__ == "__main__":
    test()
