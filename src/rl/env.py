#!/usr/bin/env python3
import gym
import numpy as np
import pandas as pd
from pathlib import Path

class RobotEnv(gym.Env):
    def __init__(self, csv_path, walls):
        super(RobotEnv, self).__init__()
        self.csv_path = Path(csv_path)
        self.files = sorted(self.csv_path.glob("*.csv"))
        self.current_episode = 0
        self.walls = walls

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # 바퀴 가속도
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # x, y, yaw

        self.goal_pos = np.array([25.0, 0.0])
        self.reset()

    def reset(self):
        if self.current_episode >= len(self.files):
            raise StopIteration("episode ended")

        self.data = pd.read_csv(self.files[self.current_episode])
        self.current_step = 0
        self.robot_pos = np.array([-25.0, 0.0])
        self.robot_yaw = 0.0
        self.current_episode += 1
        return self._get_observation()

    def step(self, action):
        left_acc, right_acc = action
        delta_pos = np.array([left_acc + right_acc, left_acc - right_acc]) * 0.05
        self.robot_pos += delta_pos
        self.current_step += 1

        reward, done = self._calculate_rewards()

        if self.current_step >= len(self.data) or done:
            return self._get_observation(), reward, True, {}

        return self._get_observation(), reward, False, {}

    def _get_observation(self):
        return np.concatenate([self.robot_pos, [self.robot_yaw]])

    def _calculate_rewards(self):
        reward = 0
        done = False
        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)

        # + reward
        if distance_to_goal < 0.5:
            reward += 100
            done = True
        else:
            reward += (5.0 * (30.0 - distance_to_goal))

        

        # - reward
        if self._check_collision():
            reward -= 50
            done = True

        if self._out_of_bounds():
            reward -= 50
            done = True

        return reward, done

    def _check_collision(self):
        x, y = self.robot_pos
        return any(wall[0] <= x <= wall[2] and wall[1] <= y <= wall[3] for wall in self.walls)

    def _out_of_bounds(self):
        x, y = self.robot_pos
        return not (-30 <= x <= 30 and -10 <= y <= 10)

    def render(self, mode="human"):
        pass
