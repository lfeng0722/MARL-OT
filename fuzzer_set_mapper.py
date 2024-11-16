import numpy as np

def map_to_vehicle_coordinates(theta, movement_vector):
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    movement = np.array(movement_vector)

    vehicle_movement = R.dot(movement)

    dx_vehicle, dy_vehicle = vehicle_movement
    return dx_vehicle, dy_vehicle

def FSM_mapper(output_action, heading):
    x, y = map_to_vehicle_coordinates(heading, output_action)
    car_action = [1, 3]

    if x > 0:
        car_action[0] = 0
    elif x < 0:
        car_action[0] = 2
    if y < 0:
        car_action[1] = 0
    return car_action