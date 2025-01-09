#!/usr/bin/env python3
import matplotlib.pyplot as plt

def plot_episode_rewards(rewards, filename):
    plt.figure()
    plt.plot(rewards)
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(filename)
    plt.close()
