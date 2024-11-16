import math
import random
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.lange_change_policy import LaneChangePolicy
import numpy as np
def determine_position_with_heading(npc_x, npc_y, ego_x, ego_y, ego_heading, lane_width=3.5):
    """
    根据 NPC 车辆和 Ego 车辆的全局坐标及 Ego 车辆的航向角，分类 NPC 车辆相对于 Ego 车辆的位置。

    参数:
    - npc_x (float): NPC 车辆的全局 X 坐标。
    - npc_y (float): NPC 车辆的全局 Y 坐标。
    - ego_x (float): Ego 车辆的全局 X 坐标。
    - ego_y (float): Ego 车辆的全局 Y 坐标。
    - ego_heading (float): Ego 车辆的航向角（弧度），范围在 [-π, π]。
    - lane_width (float, optional): 车道宽度，默认为 3.5 米。

    返回:
    - classification (str): 分类结果，可能的值包括：
        'ahead', 'left front', 'right front',
        'behind', 'left behind', 'right behind'
    """
    # 1. 坐标转换：将 NPC 车辆的全局坐标转换到 Ego 车辆的局部坐标系中
    # 计算相对位置
    dx = npc_x - ego_x
    dy = npc_y - ego_y
    distance_threshold = lane_width*1.414
    distance = np.sqrt(dx ** 2 + dy ** 2)
    # print(distance)

    # 3. 距离判断
    if distance > distance_threshold:
        return 'unknown'
    # 构建旋转矩阵（逆时针旋转 -theta）
    cos_theta = np.cos(-ego_heading)
    sin_theta = np.sin(-ego_heading)
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # 相对位置向量
    relative_pos = np.array([dx, dy])

    # 应用旋转矩阵
    local_pos = R.dot(relative_pos)
    x_local, y_local = local_pos

    # 2. 区域分类
    half_lane = lane_width

    if x_local > 0:
        if abs(y_local) <= half_lane:
            classification = 'ahead'
        elif y_local > half_lane:
            classification = 'left front'
        else:  # y_local < -half_lane
            classification = 'right front'
    elif x_local < 0:
        if abs(y_local) <= half_lane:
            classification = 'behind'
        elif y_local > half_lane:
            classification = 'left behind'
        else:  # y_local < -half_lane
            classification = 'right behind'
    else:
        # 当 x_local == 0 时，根据 y_local 分类
        if y_local > half_lane:
            classification = 'left front'  # 视为前方左侧
        elif y_local < -half_lane:
            classification = 'right front'  # 视为前方右侧
        else:
            classification = 'ahead'  # 视为前方同一车道

    return classification


def ahead(vehicle, env):
    R = random.uniform(0, 1)
    if R<1/3:
        # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
        action = [[0,0.01],[0,0.01],[0,0.01]]
        return action
    if R>=1/3 and R<2/3:
        # env.engine.add_policy(vehicle.id, LaneChangePolicy, vehicle, env.engine.generate_seed())
        first_number = random.choice([0, 2])
        second_number = 2 if first_number == 0 else 0
        action = [[first_number, 3],[1,3],[1,3],[second_number, 3],[1,3],[1,3]]
        return action
    if R>=2/3:
        # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
        action = [[0, -0.99],[0, -0.99],[0, -0.99]]
        return action

def left_front(vehicle, env):
    action =[]
    # env.engine.add_policy(vehicle.id, LaneChangePolicy, vehicle, env.engine.generate_seed())
    action.append([0,3])
    action.append([1, 3])
    action.append([1, 3])
    R = random.uniform(0, 1)
    if R<1/3:
        # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
        action.append([0,0.01])
        action.append([0, 0.01])
        action.append([0, 0.01])
    if R>=1/3 and R<2/3:
        # env.engine.add_policy(vehicle.id, LaneChangePolicy, vehicle, env.engine.generate_seed())
        first_number = random.choice([0, 2])
        second_number = 2 if first_number == 0 else 0
        action.append([first_number, 3])
        action.append([1, 3])
        action.append([1, 3])

    if R>=2/3:
        # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
        action.append([0, -0.99])
        action.append([0, -0.99])
        action.append([0, -0.99])
    return action

def right_front(vehicle, env):
    action =[]
    # env.engine.add_policy(vehicle.id, LaneChangePolicy, vehicle, env.engine.generate_seed())
    action.append([2,3])
    action.append([1, 3])
    action.append([1, 3])
    R = random.uniform(0, 1)
    if R<1/3:
        # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
        action.append([0,0.01])
        action.append([0, 0.01])
        action.append([0, 0.01])
    if R>=1/3 and R<2/3:
        # env.engine.add_policy(vehicle.id, LaneChangePolicy, vehicle, env.engine.generate_seed())
        first_number = random.choice([0, 2])
        second_number = 2 if first_number == 0 else 0
        action.append([first_number, 3])
        action.append([1, 3])
        action.append([1, 3])

    if R>=2/3:
        # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
        action.append([0, -0.99])
        action.append([0, -0.99])
        action.append([0, -0.99])
    return action


def behind(vehicle, ego, env):
    safety_th = 5

    dx = vehicle.position[0] - ego.position[0]
    dy = vehicle.position[1] - ego.position[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)

    if distance>safety_th:
        # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
        action = [[0, 0.99], [0, 0.99], [0, 0.99]]
        return action
    else:
        # env.engine.add_policy(vehicle.id, LaneChangePolicy, vehicle, env.engine.generate_seed())
        first_number = random.choice([0, 2])
        action = [[first_number, 3],[1,3],[1,3]]
        return action

def side_behind(vehicle, env):
    # env.engine.add_policy(vehicle.id, IDMPolicy, vehicle, env.engine.generate_seed())
    action = [[0, 0.99], [0, 0.99], [0, 0.99]]
    return action

def ego_stuck(points):
    if len(points) == 100:
        distance = math.sqrt((points[0][0] - points[-1][0]) ** 2 + (points[0][1] - points[-1][1]) ** 2)
        if distance < 2:
            return True

    else:
        return False
def ego_reverse(headings, tolerance=np.pi/2):
    """
       检测是否存在倒车行为。
       输入：headings（列表）：10个frame的heading，弧度制，范围[-pi, pi]。
       tolerance（float）：容忍度，默认为pi/2，表示允许一定范围内的误差。

       输出：True表示检测到倒车行为，False表示没有检测到。
       """
    if len(headings) == 5:
    # 判断两个heading是否在容忍范围内相反
        def is_opposite(heading1, heading2, tolerance):
            diff = (heading2 - heading1 + np.pi) % (2 * np.pi) - np.pi  # 归一化到[-pi, pi]范围
            return np.abs(diff) >= np.pi - tolerance and np.abs(diff) <= np.pi + tolerance

        # 检查后续9个heading是否都与第一个heading方向相反（在容忍范围内）
        for i in range(1, len(headings)):  # 从第一个frame开始，检查后续9个frame
            if all(is_opposite(headings[0], headings[j], tolerance) for j in range(1, i + 1)):
                return True  # 如果后9个heading与第一个heading相反，判断为倒车
            else:
                return False
    return False  # 如果没有检测到倒车，返回False