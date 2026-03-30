import numpy as np

# Defining the grid size
rows, cols = 10, 10

# Creating a grid initialized with zeros (free cells)
grid = np.zeros((rows, cols), dtype=int)

# Defining terrain costs and obstacles
# Example: 1 for normal terrain, 5 for rough terrain, 9 for obstacle
# Setting some rough terrain cells
grid[2:5, 3] = 5
grid[6, 1:4] = 5

# Setting some static obstacles
grid[0, 5] = 9
grid[4, 7] = 9
grid[7, 8] = 9

# Dynamic obstacles represented with a separate dictionary or list
# Store their current positions and movement patterns
dynamic_obstacles = {
    "vehicle1": {"position": (1, 1), "path": [(1, 1), (1, 2), (2, 2), (2, 1)], "step": 0},
    "vehicle2": {"position": (8, 5), "path": [(8, 5), (7, 5), (6, 5), (7, 5)], "step": 0}
}

def move_dynamic_obstacles(obstacles):
    for obs in obstacles.values():
        obs["step"] = (obs["step"] + 1) % len(obs["path"])
        obs["position"] = obs["path"][obs["step"]]

# Print grid with dynamic obstacles for visualization
def print_grid(grid, dynamic_obstacles):
    display_grid = grid.copy()
    for obs in dynamic_obstacles.values():
        x, y = obs["position"]
        display_grid[x, y] = 7  # Assign a distinct value for dynamic obstacle
    for row in display_grid:
        print(" ".join(str(cell) for cell in row))

# Example usage
print("Initial grid:")
print_grid(grid, dynamic_obstacles)

# Move dynamic obstacles and print grid again
move_dynamic_obstacles(dynamic_obstacles)
from collections import deque

def bfs(grid, start, goal):
    rows, cols = grid.shape
    visited = set()
    queue = deque([(start, [start])])  # queue of (current_node, path_to_node)

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path

        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:  # 4-connected moves
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != 9 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    return None
import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start,goal), 0, start, [start]))
    visited = {}

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path

        if current in visited and visited[current] <= cost:
            continue
        visited[current] = cost

        x, y = current
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != 9:
                new_cost = cost + grid[nx, ny]
                new_pos = (nx, ny)
                heapq.heappush(open_set, (new_cost + heuristic(new_pos, goal), new_cost, new_pos, path + [new_pos]))
    return None
def hill_climbing(grid, start, goal):
    current = start
    path = [start]

    while current != goal:
        neighbors = []
        x, y = current
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] != 9:
                neighbors.append((nx, ny))

        # Choose neighbor with lowest heuristic cost to goal
        next_cell = min(neighbors, key=lambda cell: heuristic(cell, goal), default=None)
        if next_cell is None or heuristic(next_cell, goal) >= heuristic(current, goal):
            # No better moves found - stuck in local maximum
            break

        current = next_cell
        path.append(current)

    return path if current == goal else None
start = (0, 0)
goal = (7, 7)

path_bfs = bfs(grid, start, goal)
path_astar = astar(grid, start, goal)
path_hill = hill_climbing(grid, start, goal)

print(f"BFS path: {path_bfs}")
print(f"A* path: {path_astar}")
print(f"Hill climbing path: {path_hill}")
def is_blocked(pos, dynamic_obstacles):
    """Check if pos is occupied by any dynamic obstacle"""
    return any(obs['position'] == pos for obs in dynamic_obstacles.values())

def plan_path(grid, start, goal, method='astar'):
    if method == 'bfs':
        return bfs(grid, start, goal)
    elif method == 'astar':
        return astar(grid, start, goal)
    elif method == 'hill_climbing':
        return hill_climbing(grid, start, goal)
    else:
        raise ValueError("Unknown method")

def run_agent(grid, start, goal, dynamic_obstacles, method='astar'):
    current = start
    path = plan_path(grid, current, goal, method)
    log = []
    step_count = 0

    while current != goal and path:
        next_pos = path[1]  # next step in the path (path[0] is current)
        if is_blocked(next_pos, dynamic_obstacles):
            # Dynamic obstacle blocks next cell, replan path
            log.append(f"Replan triggered at step {step_count} due to obstacle at {next_pos}")
            path = plan_path(grid, current, goal, method)
            if path is None:
                print("No available path after replanning")
                break
        else:
            # Move agent to next position
            current = next_pos
            path = path[1:]
            step_count += 1
            # Move dynamic obstacles for next time step
            move_dynamic_obstacles(dynamic_obstacles)
            # Optional: print current grid state with obstacles and agent

    return log, current == goal

# Example run
start = (0, 0)
goal = (7, 7)

dynamic_obstacles = {
    "veh1": {"position": (1, 1), "path": [(1,1),(1,2),(2,2),(2,1)], "step": 0},
    "veh2": {"position": (8, 5), "path": [(8,5),(7,5),(6,5),(7,5)], "step": 0}
}

log, success = run_agent(grid, start, goal, dynamic_obstacles, 'astar')
for entry in log:
    print(entry)
print("Success:", success)
import numpy as np

# Small map (5x5)
map_small = np.array([
    [1,1,1,9,1],
    [1,9,1,1,1],
    [1,1,1,9,1],
    [9,1,1,1,1],
    [1,1,9,1,1]
])

# Medium map (10x10)
map_medium = np.ones((10,10), dtype=int)
map_medium[2:5,3] = 5
map_medium[0,5] = 9
map_medium[7,8] = 9
map_medium[4,7] = 9

# Large map (15x15) with more obstacles and costs
map_large = np.ones((15,15), dtype=int)
map_large[3:7, 5] = 5
map_large[10, 10] = 9
map_large[6:10, 7] = 9
map_large[12, 3:8] = 5

# Map with dynamic obstacles similar to above
map_dynamic = map_medium.copy()
dynamic_obstacles_medium = {
    "veh1": {"position": (1, 1), "path": [(1,1),(1,2),(2,2),(2,1)], "step": 0},
    "veh2": {"position": (8, 5), "path": [(8,5),(7,5),(6,5),(7,5)], "step": 0}
}
def test_agent_on_map(grid, dynamic_obstacles, start, goal, method='astar'):
    log, success = run_agent(grid, start, goal, dynamic_obstacles, method)
    print(f"Testing on map with start {start} and goal {goal} using {method}:")
    for entry in log:
        print(entry)
    print("Success:", success)
    print("-" * 40)

# Small map test without dynamic obstacles
test_agent_on_map(map_small, {}, (0,0), (4,4), method='bfs')

# Medium map with dynamic obstacles
test_agent_on_map(map_medium, dynamic_obstacles_medium, (0,0), (7,7), method='astar')

# Large map test without dynamic obstacles
test_agent_on_map(map_large, {}, (0,0), (14,14), method='astar')
import time
import pandas as pd

def path_cost(grid, path):
    return sum(grid[x,y] for x,y in path) if path else float('inf')

def run_agent_with_stats(grid, start, goal, dynamic_obstacles, method='astar'):
    import time
    start_time = time.time()
    node_expansions = 0
    replans = 0

    current = start
    path = plan_path(grid, current, goal, method)
    log = []
    step_count = 0

    while current != goal and path:
        next_pos = path[1] if len(path) > 1 else path[0]
        node_expansions += 1
        if is_blocked(next_pos, dynamic_obstacles):
            replans += 1
            log.append(f"Replan at step {step_count} due to obstacle at {next_pos}")
            path = plan_path(grid, current, goal, method)
            if path is None:
                break
        else:
            current = next_pos
            path = path[1:]
            step_count += 1
            move_dynamic_obstacles(dynamic_obstacles)

    end_time = time.time()
    total_cost = path_cost(grid, path) if path else float('inf')
    success = current == goal
    return {
        "method": method,
        "success": success,
        "time_sec": end_time - start_time,
        "nodes_expanded": node_expansions,
        "replans": replans,
        "path_cost": total_cost,
        "log": log
    }


# Collect results on several maps & methods
results = []

maps = {
    "small": map_small,
    "medium": map_medium,
    "large": map_large
}

dynamic_maps = {
    "medium": dynamic_obstacles_medium
}

start_goal = {
    "small": ((0,0),(4,4)),
    "medium": ((0,0),(7,7)),
    "large": ((0,0),(14,14))
}

methods = ['bfs', 'astar', 'hill_climbing']

for map_name, grid in maps.items():
    for method in methods:
        dyn_obs = dynamic_maps.get(map_name, {})
        res = run_agent_with_stats(grid, start_goal[map_name][0], start_goal[map_name][1], dyn_obs, method)
        res["map"] = map_name
        results.append(res)

df_results = pd.DataFrame(results)

print(df_results)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Plot path cost by method and map
plt.figure(figsize=(10, 6))
sns.barplot(data=df_results, x="map", y="path_cost", hue="method")
plt.title("Path Cost Comparison")
plt.show()

# Plot nodes expanded
plt.figure(figsize=(10, 6))
sns.barplot(data=df_results, x="map", y="nodes_expanded", hue="method")
plt.title("Nodes Expanded Comparison")
plt.show()

# Plot computation time
plt.figure(figsize=(10, 6))
sns.barplot(data=df_results, x="map", y="time_sec", hue="method")
plt.title("Time Taken Comparison")
plt.show()

# For replans - separate dynamic maps only
df_dynamic = df_results[df_results["map"]=="medium"]
plt.figure(figsize=(8,5))
sns.barplot(data=df_dynamic, x="method", y="replans")
plt.title("Replans Triggered (Dynamic Map)")
plt.show()

