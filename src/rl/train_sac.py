#!/usr/bin/env python3
import gym
from stable_baselines3 import SAC
from env import RobotEnv
from pathlib import Path

def main():
    csv_path = Path("/home/teus/rl/data/train/20_200_1000/csv")
    env = RobotEnv(csv_path, walls=[(-30, -5, 30, -5), (-30, 5, 30, 5)])
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./results/train/")
    model.learn(total_timesteps=1e6)

    model.save("./results/train/sac_robot")
    env.close()

if __name__ == "__main__":
    main()
