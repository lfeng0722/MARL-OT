import math
import random
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.lange_change_policy import LaneChangePolicy
import numpy as np
def determine_position_with_heading(npc_x, npc_y, ego_x, ego_y, ego_heading, lane_width=3.5):

    dx = npc_x - ego_x
    dy = npc_y - ego_y
    distance_threshold = lane_width*1.414
    distance = np.sqrt(dx ** 2 + dy ** 2)

    if distance > distance_threshold:
        return 'unknown'

    cos_theta = np.cos(-ego_heading)
    sin_theta = np.sin(-ego_heading)
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])


    relative_pos = np.array([dx, dy])


    local_pos = R.dot(relative_pos)
    x_local, y_local = local_pos


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

        if y_local > half_lane:
            classification = 'left front'
        elif y_local < -half_lane:
            classification = 'right front'
        else:
            classification = 'ahead'

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

    if len(headings) == 5:

        def is_opposite(heading1, heading2, tolerance):
            diff = (heading2 - heading1 + np.pi) % (2 * np.pi) - np.pi
            return np.abs(diff) >= np.pi - tolerance and np.abs(diff) <= np.pi + tolerance


        for i in range(1, len(headings)):
            if all(is_opposite(headings[0], headings[j], tolerance) for j in range(1, i + 1)):
                return True
            else:
                return False
    return False