import os
from typing import List

import numpy as np

from agent import Agent


class MazeEnvironment:
    def __init__(
        self,
        init_location: np.ndarray,
        walls: List[np.ndarray],
        exit_point: np.ndarray,
        init_heading: int=180,
        exit_range: float=5.0,
        agent_kwargs: dict={}
    ) -> None:
        self.walls = walls
        self.exit_point = exit_point
        self.exit_range = exit_range
        self.init_location: np.ndarray = init_location
        self.init_heading = init_heading
        self.agent_kwargs = agent_kwargs
        self.agent = None
        self.exit_found = None

    def reset(self):
        self.agent = Agent(location=self.init_location, heading=self.init_heading, **self.agent_kwargs)

        self.exit_found = False
        # The initial distance of agent from exit
        self.initial_distance = self.agent.distance_to_exit(self.exit_point)

        # Update sensors
        self.agent.update_rangefinder_sensors(self.walls)
        self.agent.update_radars(self.exit_point)

    def get_distance_to_exit(self):
        return self.agent.distance_to_exit(self.exit_point)

    def get_agent_location(self) -> np.ndarray:
        return self.agent.location.copy()

    def get_observation(self):
        return self.agent.get_obs()

    def test_wall_collision(self, location):

        A = self.walls[:,0,:]
        B = self.walls[:,1,:]
        C = np.expand_dims(location, axis=0)
        BA = B-A

        uTop = np.sum( (C - A) * BA, axis=1)
        uBot = np.sum(np.square(BA), axis=1)

        u = uTop / uBot

        dist1 = np.minimum(
            np.linalg.norm(A - C, axis=1),
            np.linalg.norm(B - C, axis=1))
        dist2 = np.linalg.norm(A + np.expand_dims(u, axis=-1) * BA - C, axis=1)

        distances = np.where((u<0) | (u>1), dist1, dist2)

        return np.min(distances) < self.agent.radius


    def update(self, control_signals):
        if self.exit_found:
            # Maze exit already found
            return True

        # Apply control signals
        self.agent.apply_control_signals(control_signals)

        # get X and Y velocity components
        vel = np.array([np.cos(self.agent.heading/180*np.pi) * self.agent.speed,
                        np.sin(self.agent.heading/180*np.pi) * self.agent.speed])

        # Update current Agent's heading (we consider the simulation time step size equal to 1s
        # and the angular velocity as degrees per second)
        self.agent.heading = (self.agent.heading + self.agent.angular_vel) % 360

        # find the next location of the agent
        new_loc = self.agent.location + vel

        if not self.test_wall_collision(new_loc):
            self.agent.location = new_loc

        # update agent's sensors
        self.agent.update_rangefinder_sensors(self.walls)
        self.agent.update_radars(self.exit_point)

        # check if agent reached exit point
        distance = self.get_distance_to_exit()
        self.exit_found = (distance < self.exit_range)

        return self.exit_found

    @classmethod
    def read_environment(
        cls,
        ROOT_DIR: str,
        maze_name: str,
        maze_kwargs: dict={},
        agent_kwargs: dict={}
    ) -> "MazeEnvironment":
        """
        The function to read maze environment configuration from provided
        file.
        Arguments:
            file_path: The path to the file to read maze configuration from.
        Returns:
            The initialized maze environment.
        """
        maze_file = os.path.join(ROOT_DIR, 'envs', 'maze', 'maze_files', f'{maze_name}.txt')

        index = 0
        walls: List[np.ndarray] = []
        maze_exit = None
        with open(maze_file, 'r') as file:
            for line in file.readlines():
                line = line.strip()
                if len(line) == 0:
                    # skip empty lines
                    continue

                elif index == 0:
                    # read the agent's position
                    loc = np.array(list(map(float, line.split(' '))))
                elif index == 1:
                    # read the maze exit location
                    maze_exit = np.array(list(map(float, line.split(' '))))
                else:
                    # read the walls
                    wall = np.array(list(map(float, line.split(' '))))
                    walls.append(wall)

                # increment cursor
                index += 1

        walls = np.reshape(np.vstack(walls), (-1,2,2))

        # create and return the maze environment
        return MazeEnvironment(
            init_location=loc,
            walls=walls,
            exit_point=maze_exit,
            **maze_kwargs,
            agent_kwargs=agent_kwargs
        )
