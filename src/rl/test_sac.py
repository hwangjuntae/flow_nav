#!/usr/bin/env python3
import gym
from stable_baselines3 import SAC
from env import RobotEnv
from pathlib import Path

def main():
    model_path = "./results/train/sac_robot.zip"
    csv_path = Path("/home/teus/rl/data/test/20_200_1000/csv")
    env = RobotEnv(csv_path, walls=[(-30, -5, 30, -5), (-30, 5, 30, 5)])
    model = SAC.load(model_path)

    results = {"success": 0, "collision": 0, "timeout": 0, "idle": 0}

    try:
        while True:
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

            # Log results (example condition)
            results["success"] += 1  # Replace with real logic
    except StopIteration:
        pass

    print(results)
    env.close()

if __name__ == "__main__":
    main()
