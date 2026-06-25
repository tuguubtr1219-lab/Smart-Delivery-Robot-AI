import heapq
import cv2
import numpy as np

import app_state as st
from config import *
from lidar_safety import get_lidar_zone_name_for_pixel, mark_lidar_zone_obstacle


def reset_local_map():
    st.local_occ[:] = 0
    st.local_free[:] = 0
    st.local_occ_hits[:] = 0


def reset_global_map():
    st.global_occ[:] = 0
    st.global_free[:] = 0
    st.global_occ_hits[:] = 0


def in_grid(x, y, w, h):
    return 0 <= x < w and 0 <= y < h


def local_world_to_grid(lateral_m, forward_m):
    gx = int(LOCAL_ROBOT_CELL[0] + lateral_m / LOCAL_GRID_RESOLUTION_M)
    gy = int(LOCAL_ROBOT_CELL[1] - forward_m / LOCAL_GRID_RESOLUTION_M)
    return gx, gy


def global_world_to_grid(x_m, y_m):
    gx = int(GLOBAL_ORIGIN_CELL[0] + x_m / GLOBAL_GRID_RESOLUTION_M)
    gy = int(GLOBAL_ORIGIN_CELL[1] - y_m / GLOBAL_GRID_RESOLUTION_M)
    return gx, gy


def ray_cells(x0, y0, x1, y1, w, h):
    if not in_grid(x0, y0, w, h) or not in_grid(x1, y1, w, h):
        return []

    cells = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0

    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        if not in_grid(x, y, w, h):
            break
    return cells


def update_local_ray(lateral_m, forward_m):
    gx, gy = local_world_to_grid(lateral_m, forward_m)
    x0, y0 = LOCAL_ROBOT_CELL
    cells = ray_cells(x0, y0, gx, gy, LOCAL_GRID_W, LOCAL_GRID_H)
    if len(cells) <= 1:
        return

    for x, y in cells[:-1]:
        st.local_free[y, x] = min(FREE_MAX, st.local_free[y, x] + FREE_HIT_INC)
        st.local_occ[y, x] = max(0, st.local_occ[y, x] - OCC_FREE_PENALTY)

    ex, ey = cells[-1]
    st.local_occ[ey, ex] = min(OCC_MAX, st.local_occ[ey, ex] + OCC_HIT_INC)
    st.local_free[ey, ex] = max(0, st.local_free[ey, ex] - FREE_OCC_PENALTY)


def clear_local_false_obstacle(gx, gy, radius=FALSE_OBSTACLE_CLEAR_RADIUS):
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx = gx + dx
            ny = gy + dy
            if in_grid(nx, ny, LOCAL_GRID_W, LOCAL_GRID_H) and dx * dx + dy * dy <= radius * radius:
                st.local_occ[ny, nx] = max(0, st.local_occ[ny, nx] - FREE_CLEAR_STRONG_DEC)
                st.local_free[ny, nx] = min(FREE_MAX, st.local_free[ny, nx] + FREE_HIT_INC)
                st.local_occ_hits[ny, nx] = 0


def clear_global_false_obstacle(gx, gy, radius=FALSE_OBSTACLE_CLEAR_RADIUS):
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx = gx + dx
            ny = gy + dy
            if in_grid(nx, ny, GLOBAL_GRID_W, GLOBAL_GRID_H) and dx * dx + dy * dy <= radius * radius:
                st.global_occ[ny, nx] = max(0, st.global_occ[ny, nx] - FREE_CLEAR_STRONG_DEC)
                st.global_free[ny, nx] = min(FREE_MAX, st.global_free[ny, nx] + FREE_HIT_INC)
                st.global_occ_hits[ny, nx] = 0


def clear_area_around_robot(robot_pose=None):
    rx, ry = LOCAL_ROBOT_CELL
    clear_local_false_obstacle(rx, ry, radius=ROBOT_CLEAR_RADIUS_CELLS)
    if robot_pose is not None:
        gx, gy = global_world_to_grid(robot_pose[0], robot_pose[1])
        if in_grid(gx, gy, GLOBAL_GRID_W, GLOBAL_GRID_H):
            clear_global_false_obstacle(gx, gy, radius=ROBOT_CLEAR_RADIUS_CELLS)


def update_local_free_ray_only(lateral_m, forward_m, free_ratio=MAP_CLOSE_FREE_RAY_RATIO):
    safe_forward = max(MIN_DEPTH_M, forward_m * free_ratio)
    safe_lateral = lateral_m * free_ratio
    gx, gy = local_world_to_grid(safe_lateral, safe_forward)
    x0, y0 = LOCAL_ROBOT_CELL
    cells = ray_cells(x0, y0, gx, gy, LOCAL_GRID_W, LOCAL_GRID_H)
    if len(cells) <= 1:
        return
    for x, y in cells:
        st.local_free[y, x] = min(FREE_MAX, st.local_free[y, x] + FREE_HIT_INC)
        st.local_occ[y, x] = max(0, st.local_occ[y, x] - FREE_CLEAR_STRONG_DEC)
        st.local_occ_hits[y, x] = 0


def update_global_free_ray_only(lateral_m, forward_m, robot_pose, free_ratio=MAP_CLOSE_FREE_RAY_RATIO):
    safe_forward = max(MIN_DEPTH_M, forward_m * free_ratio)
    safe_lateral = lateral_m * free_ratio
    world_x, world_y = local_to_global(safe_lateral, safe_forward, robot_pose)
    robot_x, robot_y, _ = robot_pose
    x0, y0 = global_world_to_grid(robot_x, robot_y)
    x1, y1 = global_world_to_grid(world_x, world_y)
    cells = ray_cells(x0, y0, x1, y1, GLOBAL_GRID_W, GLOBAL_GRID_H)
    if len(cells) <= 1:
        return
    for x, y in cells:
        st.global_free[y, x] = min(FREE_MAX, st.global_free[y, x] + FREE_HIT_INC)
        st.global_occ[y, x] = max(0, st.global_occ[y, x] - FREE_CLEAR_STRONG_DEC)
        st.global_occ_hits[y, x] = 0


def add_global_obstacle(lateral_m, forward_m, robot_pose, radius_cells=1):
    world_x, world_y = local_to_global(lateral_m, forward_m, robot_pose)
    gx, gy = global_world_to_grid(world_x, world_y)
    if not in_grid(gx, gy, GLOBAL_GRID_W, GLOBAL_GRID_H):
        return
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            nx = gx + dx
            ny = gy + dy
            if in_grid(nx, ny, GLOBAL_GRID_W, GLOBAL_GRID_H) and dx * dx + dy * dy <= radius_cells * radius_cells:
                st.global_occ_hits[ny, nx] = min(255, int(st.global_occ_hits[ny, nx]) + 1)
                if st.global_occ_hits[ny, nx] >= OBSTACLE_CONFIRM_HITS:
                    st.global_occ[ny, nx] = min(OCC_MAX, st.global_occ[ny, nx] + OCC_HIT_INC)
                    st.global_free[ny, nx] = max(0, st.global_free[ny, nx] - FREE_OCC_PENALTY)
                else:
                    st.global_occ[ny, nx] = min(OCC_MAX, st.global_occ[ny, nx] + OCC_HIT_INC * 0.35)


def local_to_global(lateral_m, forward_m, robot_pose):
    rx, ry, theta = robot_pose
    gx = rx + lateral_m * np.cos(theta) + forward_m * np.sin(theta)
    gy = ry + forward_m * np.cos(theta) - lateral_m * np.sin(theta)
    return gx, gy


def add_local_obstacle(lateral_m, forward_m, radius_cells=1):
    gx, gy = local_world_to_grid(lateral_m, forward_m)
    if not in_grid(gx, gy, LOCAL_GRID_W, LOCAL_GRID_H):
        return
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            nx = gx + dx
            ny = gy + dy
            if in_grid(nx, ny, LOCAL_GRID_W, LOCAL_GRID_H) and dx * dx + dy * dy <= radius_cells * radius_cells:
                st.local_occ_hits[ny, nx] = min(255, int(st.local_occ_hits[ny, nx]) + 1)
                if st.local_occ_hits[ny, nx] >= OBSTACLE_CONFIRM_HITS:
                    st.local_occ[ny, nx] = min(OCC_MAX, st.local_occ[ny, nx] + OCC_HIT_INC)
                    st.local_free[ny, nx] = max(0, st.local_free[ny, nx] - FREE_OCC_PENALTY)
                else:
                    st.local_occ[ny, nx] = min(OCC_MAX, st.local_occ[ny, nx] + OCC_HIT_INC * 0.35)


def update_map_from_depth(depth_resized, frame_w, frame_h, step, robot_pose=None, update_global=False):
    if depth_resized is None:
        return

    st.local_occ[:] *= OCC_DECAY
    st.local_free[:] *= FREE_DECAY
    if update_global:
        st.global_occ[:] *= OCC_DECAY
        st.global_free[:] *= FREE_DECAY

    clear_area_around_robot(robot_pose if update_global else None)

    y_start = int(frame_h * 0.20)
    y_end = int(frame_h * 0.95)
    obstacle_y_min = int(frame_h * OBSTACLE_MARK_Y_MIN)
    obstacle_y_max = int(frame_h * OBSTACLE_MARK_Y_MAX)
    floor_ignore_y = int(frame_h * FLOOR_IGNORE_Y_RATIO)

    for y in range(y_start, y_end, step):
        row = depth_resized[y]
        for x in range(0, frame_w, step):
            z = float(row[x])
            if not np.isfinite(z) or z < MIN_DEPTH_M or z > MAX_DEPTH_M:
                continue

            normalized_x = (x - frame_w / 2.0) / (frame_w / 2.0)
            lateral_m = normalized_x * z * LATERAL_SCALE
            forward_m = z

            update_local_free_ray_only(lateral_m, forward_m)
            if update_global and robot_pose is not None:
                update_global_free_ray_only(lateral_m, forward_m, robot_pose)

            likely_floor = y >= floor_ignore_y
            likely_obstacle_by_region = obstacle_y_min <= y <= obstacle_y_max
            likely_obstacle_by_distance = z <= NEAR_OBSTACLE_MARK_M
            should_mark_obstacle = (not likely_floor) and (likely_obstacle_by_region or likely_obstacle_by_distance)
            if not should_mark_obstacle:
                continue

            if z < WARNING_DISTANCE_M:
                zone_name = get_lidar_zone_name_for_pixel(x, y, frame_w, frame_h)
                mark_lidar_zone_obstacle(zone_name, z)

            add_local_obstacle(lateral_m, forward_m, radius_cells=1)
            if update_global and robot_pose is not None:
                add_global_obstacle(lateral_m, forward_m, robot_pose, radius_cells=1)


def update_map_from_yolo_bbox(x1, x2, frame_w, lidar_distance_m, is_person, robot_pose=None, update_global=False):
    if lidar_distance_m is None:
        return

    cx = (x1 + x2) / 2.0
    normalized_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
    lateral_m = normalized_x * lidar_distance_m * LATERAL_SCALE
    forward_m = lidar_distance_m

    add_local_obstacle(lateral_m, forward_m, radius_cells=4 if is_person else 3)
    if update_global and robot_pose is not None:
        add_global_obstacle(lateral_m, forward_m, robot_pose, radius_cells=4 if is_person else 3)


def get_map_layers(occ, free, inflation_radius):
    occupied = (occ >= OCC_THRESHOLD).astype(np.uint8)
    free_grid = ((free >= FREE_THRESHOLD) & (occupied == 0)).astype(np.uint8)
    kernel_size = inflation_radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    inflated = cv2.dilate(occupied, kernel, iterations=1)
    return free_grid, occupied, inflated


def astar(obstacle_grid, start, goal):
    h, w = obstacle_grid.shape

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
            nx = current[0] + dx
            ny = current[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if obstacle_grid[ny, nx] == 1:
                continue
            step_cost = 1.4 if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + step_cost
            neighbor = (nx, ny)
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
    return []


def draw_grid_map(free_grid, occupied_grid, inflated_grid, path, robot_cell, goal_cell, scale=3, title=""):
    h, w = occupied_grid.shape
    map_img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    map_img[:] = (25, 25, 25)

    free_big = cv2.resize((free_grid * 255).astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    map_img[free_big > 0] = (55, 80, 55)

    inflated_big = cv2.resize((inflated_grid * 255).astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    map_img[inflated_big > 0] = (50, 50, 120)

    occ_big = cv2.resize((occupied_grid * 255).astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    map_img[occ_big > 0] = (40, 40, 230)

    for x, y in path:
        cv2.rectangle(map_img, (x * scale, y * scale), ((x + 1) * scale - 1, (y + 1) * scale - 1), (0, 230, 255), -1)

    rx, ry = robot_cell
    gx, gy = goal_cell
    cv2.circle(map_img, (rx * scale, ry * scale), 6, (255, 255, 255), -1)
    cv2.circle(map_img, (gx * scale, gy * scale), 6, (0, 255, 0), -1)

    if title:
        cv2.putText(map_img, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return map_img
