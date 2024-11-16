import random
import time
import base64
import requests
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
from collections import deque
from metadrive.utils import print_source
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.policy.idm_policy import IDMPolicy, TrajectoryIDMPolicy
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.utils import generate_gif
from metadrive.scenario.parse_object_state import parse_object_state, get_idm_route, get_max_valid_indicis
from metadrive.component.lane.straight_lane import StraightLane
# env=MetaDriveEnv(dict(map="S", traffic_density=0))
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
import copy
import json
from MADDPG import MADDPG
from PIL import Image as im
from action_pattern import determine_position_with_heading, ahead, side_behind, behind, left_front, right_front, ego_stuck, ego_reverse
import io
import numpy as np
import itertools
import torch
from road_network import Lane
from fuzzer_set_mapper import FSM_mapper
import os
def position_scaler(position, x_min, x_max, y_min, y_max):
    # Example values for min and max positions

    # Original positions
    npc_position_x, npc_position_y = position

    # Scaling to [0, 2]
    x_range = x_max - x_min
    y_range = y_max - y_min
    scaled_x = ((npc_position_x - x_min) / x_range) * 2
    # print(min_x)
    scaled_y = ((npc_position_y - y_min) / y_range) * 2

    return scaled_x,scaled_y

violation = 0
stuck = 0
reverse = 0
crash = 0

policy =IDMPolicy
#init agent
max_x_diff = 140
max_y_diff = 140
n_agents= 4
n_actions = 2
actor_dims = [10, 10, 10, 7]
critic_dims = sum(actor_dims)
maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128,
                           alpha=0.00001, beta=0.02, scenario='MARL',
                           chkpt_dir='')
maddpg_agents.load_checkpoint()
TOPK=0

for m in range(200):


    # map_config = {BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
    #               BaseMap.GENERATE_CONFIG: 3,
    #               BaseMap.LANE_WIDTH: 3.5,
    #               BaseMap.LANE_NUM: 4}
    # map_config["config"]=5
    # env=MultiAgentMetaDrive(dict(
    #     num_scenarios=10,
    #     map_config= map_config ,
    #                        traffic_density=0,
    #                        is_multi_agent = True,
    #                        num_agents = 4,
    #                        discrete_action=True,
    #                        use_multi_discrete=True,
    #                        agent_policy = LaneChangePolicy,
    #                        random_spawn_lane_index = True,
    #                        use_render=not os.getenv('TEST_DOC'),
    #                        agent_configs={"agent0": dict(use_special_color=True, enable_reverse=True),
    #                                     #   "agent1": dict(spawn_lane_index=(spawn_point[1][0], spawn_point[1][1], 0)),
    #                                     # "agent2": dict(spawn_lane_index=(spawn_point[2][0], spawn_point[2][1], 0)),
    #                                     # "agent3": dict(spawn_lane_index=(spawn_point[3][0], spawn_point[3][1], 0)),
    #                                       }
    #                     ))
    env = MultiAgentMetaDrive(dict(
        map_config=dict(config='C', type="block_sequence", lane_num=4),
        traffic_density=0,
        is_multi_agent=True,
        num_agents=4,
        discrete_action=True,
        use_multi_discrete=True,
        agent_policy=LaneChangePolicy,
        use_render=not os.getenv('TEST_DOC'),
        random_spawn_lane_index=True,
        agent_configs={"agent0": dict(use_special_color=True, enable_reverse=True),
                       #   "agent1": dict(spawn_lane_index=(spawn_point[1][0], spawn_point[1][1], 0)),
                       # "agent2": dict(spawn_lane_index=(spawn_point[2][0], spawn_point[2][1], 0)),
                       # "agent3": dict(spawn_lane_index=(spawn_point[3][0], spawn_point[3][1], 0)),
                       }
    ))
    env.reset()
    xmin, xmax, ymin, ymax=env.current_map.get_boundary_point()





    all_vehicles = env.engine.get_objects()
    vehicles = []
    for k, v in all_vehicles.items():
        vehicles.append(v)

    ego_vehicle = vehicles[0]

    n1 = vehicles[1]
    n2 = vehicles[2]
    n3 = vehicles[3]

    frames = []

    n1_action = [0, 0]
    n2_action  = [0, 0]
    n3_action  = [0, 0]


    env.engine.add_policy(ego_vehicle.id, policy, ego_vehicle, env.engine.generate_seed())
    agents = [n1, n2, n3]  # List of the three agents (n1, n2, n3)
    agent_action = [n1_action, n2_action, n3_action]

    action_arrs = [[], [], []]
    crash_tri = False
    # reverse_tri = False
    stuck_tri = False
    eog_monitor = deque(maxlen=100)
    # eog_reverse = deque(maxlen=50)
    for _ in range(500):

        # Loop to generate states for each agent

        states = []
        trigger_arr = []
        for i, agent in enumerate(agents):
        # Compute theta and distance d relative to the ego vehicle

            trigger = determine_position_with_heading(agent.position[0], agent.position[1], ego_vehicle.position[0], ego_vehicle.position[1], ego_vehicle.heading_theta)
            trigger_arr.append(trigger)
            theta = np.arctan2(ego_vehicle.position[1] - agent.position[1],
                               ego_vehicle.position[0] - agent.position[0])
            d = np.linalg.norm(agent.last_position - ego_vehicle.last_position)

            # Calculate s_target as the normalized distance and theta
            s_target = [d / np.linalg.norm(max_x_diff+max_y_diff), theta]
            other_agent =  [(other_agent.position[0], other_agent.position[1])
                    for j, other_agent in enumerate(agents) if j != i  and j != n_agents - 1]

            # print(len(other_agent))
            # Prepare the state for the current agent
            agent_position_x , agent_position_y = position_scaler(agent.position,xmin, xmax, ymin, ymax)
            # print(agent.position)
            # print(agent.position)
            other_position1_x,other_position1_y = position_scaler(other_agent[0],xmin, xmax, ymin, ymax)
            other_position2_x, other_position2_y = position_scaler(other_agent[1],xmin, xmax, ymin, ymax)

            agent_state = (
                agent_position_x / max_x_diff,
                agent_position_y / max_y_diff,
                agent_action[i][0]*10,
                agent_action[i][1]*10,  # Current agent's position and action
                other_position1_x/ max_x_diff,
                other_position1_y/ max_y_diff,
                other_position2_x/ max_x_diff,
                other_position2_y/ max_y_diff,
                s_target[0],
                s_target[1]
            )

            states.append(agent_state)
            # print(states)
        actions = maddpg_agents.choose_action(states)
        # print(actions)
        # print(ego_vehicle.heading_theta)
        agent_action[0] += actions[0]
        agent_action[1] += actions[1]
        agent_action[2] += actions[2]

        #MARL
        n1_action = FSM_mapper(actions[0],n1.heading_theta)
        n2_action = FSM_mapper(actions[1],n2.heading_theta)
        n3_action = FSM_mapper(actions[2],n3.heading_theta)
        MARL_action = [n1_action, n2_action, n3_action ]


        p = env.engine.get_policy(ego_vehicle.name)

        try:
            # Call `p.act(True)` and get the first element from each call
            a0 = [p.act(True)[0], p.act(True)[0]]
        except Exception as e:
            # If an error occurs, print it or handle it as needed
            # print(f"Error calling p.act(True): {e}")
            a0 = [0,0] # Or set a default value if needed

        actions = {
            'agent0': a0,
            'agent1': None,
            'agent2': None,
            'agent3': None
        }

        # 遍历 agents 列表
        for i, agent in enumerate(agents):
            # print(f"agent{i+1}",action_arrs[i])
            if trigger_arr[i] != "ahead" and not action_arrs[i]:
                action_arrs[i].append(MARL_action[i])
                # actions[f'agent{i+1}'] = MARL_action[i]  # 直接更新动作
            elif trigger_arr[i] == "ahead" and not action_arrs[i]:
                # if not action_arrs[i]:
                action_p = ahead(agent, env)
                action_arrs[i] = action_p
            elif trigger_arr[i] == "left_front" and not action_arrs[i]:
                action_p = left_front(agent, env)
                action_arrs[i] = action_p
            elif trigger_arr[i] == "right_front" and not action_arrs[i]:
                action_p = right_front(agent, env)
                action_arrs[i] = action_p
            elif trigger_arr[i] == "behind" and not action_arrs[i]:
                action_p = behind(agent, ego_vehicle, env)
                action_arrs[i] = action_p
            elif trigger_arr[i] == 'left behind' or trigger_arr[i] == 'right behind' and not action_arrs[i]:
                action_p = side_behind(agent, env)
                action_arrs[i] = action_p
            actions[f'agent{i + 1}'] = action_arrs[i].pop(0)

            # print(actions)
        # Initialize the multi_actions dictionary
        for i, agent in enumerate(agents):
            dx = agent.position[0] - ego_vehicle.position[0]
            dy = agent.position[1] - ego_vehicle.position[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance<5:
                # env.engine.add_policy(agent.id, IDMPolicy, agent, env.engine.generate_seed())
                actions[f'agent{i + 1}'] = [0,-0.99]

            if isinstance(actions[f'agent{i + 1}'][1] , int):
                env.engine.add_policy(agent.id, LaneChangePolicy, agent, env.engine.generate_seed())
                # print(1111)
            else:
                env.engine.add_policy(agent.id, IDMPolicy, agent, env.engine.generate_seed())
                # print(2222)
        multi_actions = {}

        # print(env.engine.agents.keys())
    # Populate multi_actions with available agents and their actions
        for agent in env.engine.agents.keys():
            # if info[agent]["crash_vehicle"]:
            #     multi_actions[agent].pop(agent)

            if agent in actions:
                multi_actions[agent] = actions[agent]

        if 'agent0' not in env.engine.agents.keys():
            break
        _, _, tm, _, info = env.step(multi_actions)
        eog_monitor.append(ego_vehicle.position)

        frame = env.render(
            mode="topdown",
            window=False,
            screen_size=(1200, 1200),
            camera_position=(100,0))
        frames.append(frame)



        agent_dis = []
        for i, agent in enumerate(agents):
            ax = agent.position[0] - ego_vehicle.position[0]
            ay = agent.position[1] - ego_vehicle.position[1]
            a_distance = np.sqrt(ax ** 2 + ay ** 2)
            agent_dis.append(a_distance)
        d_count = sum(1 for item in agent_dis if item <= 10)
        stuck_indicator = ego_stuck(eog_monitor)

        if stuck_indicator and d_count>=2:
            stuck += 1
            violation += 1
            stuck_tri = True
            print('stuck')
            break
        if info['agent0']["crash_vehicle"] and d_count>=2:
            violation+=1
            crash+=1
            crash_tri = True
            print('crash')
        #
        if  info['agent0']["crash_vehicle"]:
            break
        if violation == 5:
            TOPK=m
    generate_gif(frames, gif_name=f"test_case/test{m}_crash.gif")
    if crash_tri:
        generate_gif(frames, gif_name=f"test_case/test{m}_crash.gif")

    if stuck_tri:
        generate_gif(frames, gif_name=f"test_case/test{m}_stuck.gif")
    env.close()
print('number of stuck:' ,stuck)
# print('number of reverse:' ,reverse)
print('number of crash:', crash)
print("violation rate:", violation/2)
print("TOP-5:", TOPK)