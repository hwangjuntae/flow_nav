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
import matplotlib.patches as patches

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

dt_time_step = 0.1  # 기본 시간 스텝 (데이터에서 실제 dt를 가져옴)
num_future_steps = 5  # 미래 이동을 예측할 스텝 수

reaction_radius = 1.0  # 반응 반경

# 로봇 초기 위치 및 방향 설정
robot_x = -25.0
robot_y = 0.0
robot_direction = 0.0  # 초기 방향 (라디안, 0은 x축 양의 방향)

# 로봇 동역학 파라미터
desired_linear_velocity = 0.0      # m/s
desired_angular_velocity = 0.0     # rad/s
current_linear_velocity = 0.0      # m/s
current_angular_velocity = 0.0     # rad/s

maxAcceleration = 2.0               # m/s²
maxAngularAcceleration = 2.68       # rad/s²
maxAngularSpeed = 3.0               # rad/s
maxDeceleration = 2.0               # m/s²
maxLinearSpeed = 7.0                # m/s
maxWheelSpeed = 0.0                  # rad/s (현재 사용되지 않음)
wheelDistance = 0.413                # m
wheelRadius = 0.14                   # m

lidar_radius = 5.0  # 로봇 라이다 반경

# 목표 위치
target_x = 25.0

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

# 초기화
fig = plt.figure(figsize=(12, 6))
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
                    dt = dt_time_step
            else:
                dt = dt_time_step

            ax_main.cla()

            # 로봇 위치 및 속도 업데이트
            # 목표 위치와의 거리 계산
            distance_to_target = target_x - robot_x

            # 간단한 제어 로직: 목표를 향해 최대 가속도로 가속, 목표 근처에서는 감속
            if distance_to_target > 0.5:
                desired_linear_velocity = maxLinearSpeed
            else:
                # 목표에 가까워지면 감속
                desired_linear_velocity = max(0.0, distance_to_target)  # 선형 속도를 줄임

            # 선형 속도 업데이트
            if current_linear_velocity < desired_linear_velocity:
                current_linear_velocity += maxAcceleration * dt
                if current_linear_velocity > desired_linear_velocity:
                    current_linear_velocity = desired_linear_velocity
            elif current_linear_velocity > desired_linear_velocity:
                current_linear_velocity -= maxDeceleration * dt
                if current_linear_velocity < desired_linear_velocity:
                    current_linear_velocity = desired_linear_velocity

            # 각속도 업데이트 (현재는 0.0으로 고정)
            desired_angular_velocity = 0.0
            if current_angular_velocity < desired_angular_velocity:
                current_angular_velocity += maxAngularAcceleration * dt
                if current_angular_velocity > desired_angular_velocity:
                    current_angular_velocity = desired_angular_velocity
            elif current_angular_velocity > desired_angular_velocity:
                current_angular_velocity -= maxAngularAcceleration * dt
                if current_angular_velocity < desired_angular_velocity:
                    current_angular_velocity = desired_angular_velocity

            # 현재 속도를 기반으로 로봇 위치 업데이트
            robot_x += current_linear_velocity * np.cos(robot_direction) * dt
            robot_y += current_linear_velocity * np.sin(robot_direction) * dt
            # 로봇의 방향 업데이트 (현재 각속도는 0)
            robot_direction += current_angular_velocity * dt
            robot_direction = robot_direction % (2 * np.pi)

            # 로봇이 목표 위치에 도달했을 때 속도 멈춤
            if distance_to_target <= 0.0:
                current_linear_velocity = 0.0

            positions = subset[['x','y']].values
            velocities = subset[['vx','vy']].values

            rho = kernel_density(positions, X, Y, bandwidth=0.5)
            VX, VY = average_velocity(positions, velocities, X, Y, radius=2.0)

            rho_future = rho.copy()
            VX_future = VX.copy()
            VY_future = VY.copy()

            for step in range(num_future_steps):
                rho_future = continuity_step(rho_future, VX_future, VY_future, dx, dy, dt)

            # 에이전트(사람) 그리기
            for idx_, agent in subset.iterrows():
                agent_x, agent_y = agent['x'], agent['y']
                direction = agent['direction']
                color = direction_to_color(direction)
                circle = plt.Circle((agent_x, agent_y), agent_radius, facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.8)
                ax_main.add_patch(circle)
                ax_main.arrow(agent_x, agent_y, agent['vx'] * 0.8, agent['vy'] * 0.8,
                              head_width=0.5, head_length=0.5, fc='black', ec='black', alpha=0.9)

            # 로봇 그리기 (정사각형)
            robot_side = 2 * agent_radius
            robot_square = patches.Rectangle((robot_x - agent_radius, robot_y - agent_radius),
                                             robot_side, robot_side,
                                             edgecolor='blue', facecolor='cyan', linewidth=1.5, alpha=0.8)
            ax_main.add_patch(robot_square)

            # 로봇에 장착된 라이다 표시(반경 5m)
            lidar_circle = plt.Circle((robot_x, robot_y), lidar_radius, color='blue', fill=False, linestyle='--', alpha=0.5)
            ax_main.add_patch(lidar_circle)

            # 로봇의 라이다 범위 내의 에이전트만 고려
            agents_in_lidar = []
            for idx_, agent in subset.iterrows():
                dist_to_robot = np.sqrt((agent['x'] - robot_x)**2 + (agent['y'] - robot_y)**2)
                if dist_to_robot <= lidar_radius:
                    agents_in_lidar.append(agent)

            # 미래 벡터 필드 화살표 (라이다 범위 내의 지점만 표시)
            for i in range(0, X.shape[0], 3):
                for j in range(0, X.shape[1], 3):
                    # 로봇 라이다 범위 내에 있는 점인지 확인
                    dist_to_robot_point = np.sqrt((X[i, j] - robot_x)**2 + (Y[i, j] - robot_y)**2)
                    if dist_to_robot_point <= lidar_radius and wall_bottom <= Y[i, j] <= wall_top:
                        # 라이다 안에 있는 에이전트들로 인해 반응하는지 확인
                        responsive = False
                        arrow_color = 'black'
                        for agent in agents_in_lidar:
                            agent_x_, agent_y_ = agent['x'], agent['y']
                            color = direction_to_color(agent['direction'])
                            if np.sqrt((X[i, j] - agent_x_)**2 + (Y[i, j] - agent_y_)**2) <= reaction_radius:
                                responsive = True
                                arrow_color = color
                                break

                        if responsive:
                            ax_main.arrow(X[i, j], Y[i, j], VX_future[i, j] * 0.5, VY_future[i, j] * 0.5,
                                          head_width=0.3, head_length=0.3, fc=arrow_color, ec=arrow_color, alpha=0.8)
                        else:
                            # 반응하지 않는 화살표
                            ax_main.arrow(X[i, j], Y[i, j], VX_future[i, j] * 0.3, VY_future[i, j] * 0.3,
                                          head_width=0.2, head_length=0.2, fc='black', ec='black', alpha=0.5)

            # 벽 표시
            ax_main.hlines(wall_top, -30, 30, colors='black', linestyles='-', linewidth=1.5)
            ax_main.hlines(wall_bottom, -30, 30, colors='black', linestyles='-', linewidth=1.5)

            ax_main.set_title(f"Time: {current_time:.2f}s (Future Probability of Movement within LiDAR range)")
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
