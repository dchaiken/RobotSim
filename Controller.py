"""
Abstract class for a controller.
"""

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from time import time, sleep
from threading import Timer
from matplotlib import cm


class Controller(ABC):

    def __init__(self, world, known, pose, vel_change_max=.1, pixel_resolution=100, fov=np.pi/3, min_vel=-1, max_vel=1, time_interval=.1):
        """
        Create an agent controller, which also maintains maps of the true world and known area
        """
        assert world.shape == known.shape
        self.world = world
        self.known = known
        # Pose should be (x, y, theta)
        self.pose = pose
        # maximum change in velocity on a timestep
        self.vel_change_max = vel_change_max
        # pixels/meter, if representing real space
        self.pixel_resolution = pixel_resolution
        # field of view size for the agent
        self.fov = max(0, min(np.pi*2, fov))
        self.min_vel = min_vel
        self.max_vel = max_vel
        # Start with 0 linear or angular velocity, units m/s
        self.velocity = [0, 0]
        self.fig = None
        self.axes = None
        self.last_control_time = time()
        self.time_interval = time_interval  # How much time each step represents
        # Save minimum distance to wall in line of sight
        self.min_dist = np.inf

    def display(self):
        """
        Display agent's location in the world and the current known map
        """
        self.fig, self.axes = plt.subplots(1, 2)
        self._draw()
        plt.show()

    def explore(self):
        """
        Update known based on what is in front of the agent in the world, assuming perfectly correct sight
        """
        x, y, theta = self.pose
        idx = np.indices((self.world.shape))

        idx[0] -= y
        idx[1] -= x
        angles = (np.arctan2(idx[0], idx[1]) + 2*np.pi) % (2*np.pi)
        low_angle = (theta - self.fov/2)
        high_angle = (theta + self.fov/2)
        # Determine area in agent field of view
        if low_angle != low_angle % (2*np.pi) or high_angle != high_angle % (2*np.pi):
            low_angle = low_angle % (2*np.pi)
            high_angle = high_angle % (2*np.pi)
            condition = np.logical_or
        else:
            condition = np.logical_and

        # Rest of function figures out where there are walls the robot can't see past
        approx_angles = np.round(angles, 2)
        # give angles in possible sight a bit of a buffer to account for rounding errors
        possible_sight = condition(
            low_angle - .01 < approx_angles, approx_angles < high_angle + .01)
        dists_from_loc = np.linalg.norm(idx, axis=0)
        seen_walls = np.logical_and(self.world == 1, possible_sight)
        walls = np.where(seen_walls)
        wall_coords = zip(walls[0], walls[1])
        # Make dict mapping angles to closest wall at that angle
        angs_to_dists = dict(sorted([(approx_angles[coord], self._points_dist(
            coord, (y, x))) for coord in wall_coords], reverse=True, key=lambda pair: pair[1]))
        # Save min distance to wall in front of agent
        self.min_dist = min(angs_to_dists.values())
        # Fill in angles between two pixels in a wall
        for wall_y, wall_x in zip(walls[0], walls[1]):
            walls_nearby = seen_walls[max(
                0, wall_y-2):wall_y+2, max(0, wall_x-2):wall_x+2]
            angles_nearby = approx_angles[max(
                0, wall_y-2):wall_y+2, max(0, wall_x-2):wall_x+2]
            min_angle_nearby = np.min(angles_nearby[walls_nearby])
            max_angle_nearby = np.max(angles_nearby[walls_nearby])
            # Case where angles wrap around 0:
            if max_angle_nearby - min_angle_nearby > self.fov:
                for wall_angle in np.arange(max_angle_nearby, 6.28+min_angle_nearby+0.01, 0.01):
                    wall_angle = int((wall_angle % 6.29) * 100)/100
                    angs_to_dists[wall_angle] = min(angs_to_dists.get(
                        wall_angle, np.inf), angs_to_dists[max_angle_nearby])
            else:
                for wall_angle in np.arange(min_angle_nearby, max_angle_nearby, 0.01):
                    wall_angle = int(wall_angle*100)/100
                    angs_to_dists[wall_angle] = min(angs_to_dists.get(
                        wall_angle, np.inf), angs_to_dists[min_angle_nearby])

        def lookup(v): return angs_to_dists.get(v, 0)
        vlookup = np.vectorize(lookup)
        dists_to_wall = vlookup(approx_angles)
        true_sight = dists_from_loc < dists_to_wall
        true_sight[np.logical_or(approx_angles == int(
            low_angle*100)/100, approx_angles == int(high_angle*100)/100)] = False
        self.known = np.where(np.invert(true_sight),
                              self.known, self.world)

    def move(self, draw=True):
        """
        Move the robot according to current velocity
        """
        self.pose[2] = (self.pose[2] + self.time_interval *
                        self.velocity[1]) % (2*np.pi)
        self.pose[1] = int(max(0, min(self.world.shape[0], self.pose[1]+self.velocity[0]
                                      * np.sin(self.pose[2])*self.time_interval*self.pixel_resolution)))
        self.pose[0] = int(max(0, min(self.world.shape[1], self.pose[0]+self.velocity[0]
                                      * np.cos(self.pose[2])*self.time_interval*self.pixel_resolution)))
        self.explore()
        if draw:
            self._draw()

    def _filter_velocity(self, desired, previous):
        """
        Use to restrict velocity and change in velocity to a reasonable range
        """
        low = max(previous - self.vel_change_max, self.min_vel)
        high = min(previous + self.vel_change_max, self.max_vel)
        return max(low, min(desired, high))

    def _points_dist(self, p1, p2):
        """
        Find euclidean distance between p1 and p2
        """
        return np.linalg.norm(np.array(p1)-np.array(p2))

    def _draw(self):
        # Theta is measured from x-axis CLOCKWISE since 0 corresponds to top array row
        plt.cla()
        plt.clf()
        self.axes = self.fig.subplots(1, 2)

        x, y, theta = self.pose
        max_y, max_x = self.world.shape
        if not (0 <= x < max_x and 0 <= y < max_y):
            raise IndexError("Out of Bounds")
        cmap = cm.get_cmap()

        circle = plt.Circle((x, y), 20, color='r')
        arrow = plt.Arrow(x, y, 100*np.cos(theta), 100 *
                          np.sin(theta), width=20, color='r')
        self.axes[0].add_artist(circle)
        self.axes[0].add_artist(arrow)
        vmin = np.min((self.world, self.known))
        vmax = np.max((self.world, self.known))
        self.axes[0].imshow(self.world, cmap=cmap, vmin=vmin,
                            vmax=vmax)
        self.axes[1].imshow(self.known, cmap=cmap, vmin=vmin,
                            vmax=vmax)
        plt.draw()

    @abstractmethod
    def control(self, event):
        """
        Abstract method that maps input to an agent velocity
        """
        print("function Controller.control not implemented")
