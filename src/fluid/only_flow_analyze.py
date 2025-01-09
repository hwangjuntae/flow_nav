#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib import colors
import matplotlib.gridspec as gridspec
import time

# CSV 파일 경로
csv_path = "/root/flow_ws/src/moving_people/src/py_social_force/csv/crowd_coordinates1.csv"

# 에이전트 반지름
agent_radius = 0.2

# 벽 범위
wall_top = 5.0
wall_bottom = -5.0

# 격자 정의
nx, ny = 100, 100
x_min, x_max = -30, 30
y_min, y_max = -10, 10
x_range = np.linspace(x_min, x_max, nx)
y_range = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x_range, y_range)

dt = 0.1  # 시간 스텝
num_future_steps = 5  # 미래 이동을 예측할 스텝 수

reaction_radius = 1.0  # 반응 반경

def direction_to_color(direction):
    normalized_direction = direction / (2 * np.pi)
    rgb = plt.cm.hsv(normalized_direction)[:3]
    return rgb

def kernel_density(positions, X, Y, bandwidth=0.5):
    density = np.zeros_like(X)
    for pos in positions:
        dx = X - pos[0]
        dy = Y - pos[1]
        dist2 = dx*dx + dy*dy
        density += np.exp(-dist2/(2*bandwidth*bandwidth))
    return density / (2*np.pi*bandwidth*bandwidth*len(positions))

def average_velocity(positions, velocities, X, Y, radius=1.0):
    VX = np.zeros_like(X)
    VY = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx = positions[:,0] - X[i,j]
            dy = positions[:,1] - Y[i,j]
            dist2 = dx*dx + dy*dy
            mask = dist2 < radius**2
            if np.any(mask):
                VX[i,j] = np.mean(velocities[mask,0])
                VY[i,j] = np.mean(velocities[mask,1])
    return VX, VY

def continuity_step(rho, VX, VY, dx, dy, dt):
    rho_new = rho.copy()
    d_rho_x = ( (rho[2:,1:-1]*VX[2:,1:-1]) - (rho[:-2,1:-1]*VX[:-2,1:-1]) )/(2*dx)
    d_rho_y = ( (rho[1:-1,2:]*VY[1:-1,2:]) - (rho[1:-1,:-2]*VY[1:-1,:-2]) )/(2*dy)

    rho_new[1:-1,1:-1] = rho[1:-1,1:-1] - dt*(d_rho_x + d_rho_y)
    rho_new[0,:] = rho_new[1,:]
    rho_new[-1,:] = rho_new[-2,:]
    rho_new[:,0] = rho_new[:,1]
    rho_new[:,-1] = rho_new[:,-2]

    rho_new[rho_new < 0] = 0
    return rho_new

fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1], wspace=0.05)

ax_main = fig.add_subplot(gs[0])
ax_colorbar = fig.add_subplot(gs[1])

sm = ScalarMappable(cmap='hsv', norm=colors.Normalize(vmin=0, vmax=2*np.pi))
sm.set_array([]) 
cbar = fig.colorbar(sm, cax=ax_colorbar)
cbar.set_label('Direction (radians)')

processed_times = set()
plt.ion()
fig.show()
fig.canvas.draw()

dx = (x_max - x_min)/(nx-1)
dy = (y_max - y_min)/(ny-1)

while True:
    try:
        data = pd.read_csv(csv_path)
        data['speed'] = np.sqrt(data['vx']**2 + data['vy']**2)
        data['direction'] = np.arctan2(data['vy'], data['vx']) 
        data['direction'] = data['direction'] % (2 * np.pi)

        unique_times = sorted(data['time'].unique())
        new_times = [t for t in unique_times if t not in processed_times]

        for current_time in new_times:
            subset = data[data['time'] == current_time]
            if len(unique_times) > 1:
                idx = unique_times.index(current_time)
                if idx < len(unique_times) - 1:
                    dt = unique_times[idx + 1] - unique_times[idx]
                else:
                    dt = 0.1
            else:
                dt = 0.1

            ax_main.cla()

            positions = subset[['x','y']].values
            velocities = subset[['vx','vy']].values

            rho = kernel_density(positions, X, Y, bandwidth=0.5)
            VX, VY = average_velocity(positions, velocities, X, Y, radius=2.0)

            rho_future = rho.copy()
            VX_future = VX.copy()
            VY_future = VY.copy()

            for step in range(num_future_steps):
                rho_future = continuity_step(rho_future, VX_future, VY_future, dx, dy, dt)

            for idx_, agent in subset.iterrows():
                ax_, ay = agent['x'], agent['y']
                direction = agent['direction']
                color = direction_to_color(direction)
                circle = plt.Circle((ax_, ay), agent_radius, color=color, edgecolor='black', linewidth=0.8, alpha=0.8)
                ax_main.add_patch(circle)
                ax_main.arrow(ax_, ay, agent['vx'] * 0.8, agent['vy'] * 0.8,
                              head_width=0.5, head_length=0.5, fc='black', ec='black', alpha=0.9)

            for i in range(0, X.shape[0], 3):  # 촘촘한 격자
                for j in range(0, X.shape[1], 3):
                    if wall_bottom <= Y[i, j] <= wall_top:  # y 값 제한
                        # 객체와 반응 여부를 확인
                        responsive = False
                        for idx_, agent in subset.iterrows():
                            ax_, ay = agent['x'], agent['y']
                            color = direction_to_color(agent['direction'])
                            if np.sqrt((X[i, j] - ax_)**2 + (Y[i, j] - ay)**2) <= reaction_radius:
                                responsive = True
                                arrow_color = color
                                break
                        if responsive:
                            ax_main.arrow(X[i, j], Y[i, j], VX_future[i, j] * 0.5, VY_future[i, j] * 0.5,
                                          head_width=0.3, head_length=0.3, fc=arrow_color, ec=arrow_color, alpha=0.8)
                        else:  # 반응하지 않은 화살표
                            ax_main.arrow(X[i, j], Y[i, j], VX_future[i, j] * 0.3, VY_future[i, j] * 0.3,
                                          head_width=0.2, head_length=0.2, fc='black', ec='black', alpha=0.5)

            ax_main.hlines(wall_top, -30, 30, colors='black', linestyles='-', linewidth=1.5)
            ax_main.hlines(wall_bottom, -30, 30, colors='black', linestyles='-', linewidth=1.5)

            ax_main.set_title(f"Time: {current_time:.2f}s (Future Probability of Movement)")
            ax_main.set_xlabel("X Position")
            ax_main.set_ylabel("Y Position")
            ax_main.axis('equal')
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(y_min, y_max)
            ax_main.grid()

            fig.canvas.draw()
            fig.canvas.flush_events()

            processed_times.add(current_time)
            time.sleep(dt)

    except KeyboardInterrupt:
        print("종료")
        break
    except Exception as e:
        print(f"오류 발생: {e}")
        time.sleep(1)
