data_path = 'examp3.txt'
lidar_angle = 240  # угол обзора лидара в градусах
threshold = 0.1  # сколько обзора в доле с каждого края закрывается самим
# роботом (положим, что лидар устновлен симметрично)
max_range = 5.6  # предел лидара
enable_frames = False  # разрешение на запись промежуточных кадров

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

df = pd.DataFrame(columns=['X', 'Y', 'W', 'Lidar_Data'])

f = open(data_path)
for line in f:
    readed_line = line
    curr_data = [readed_line[:-1].split(';')[0].split(','),
                 readed_line[:-1].split(';')[1].split(
                     ',')]  # [0] - odo, [1] - lidar
    parsed_data = []
    for i, l in enumerate(curr_data):
        # print(l,i)
        if i == 0:
            for j, el in enumerate(l):
                parsed_data.append(float(el))
        else:
            parsed_data.append([float(item) for item in l])
    # в итоге 4 ячейки: float, float, float, list
    df.loc[len(df)] = parsed_data

rot_angle = df.iloc[0]['W'] - math.pi / 2
rot_angle = math.pi / 4


def del_inf(a, d):  # удаляет элементы массива а по индексам списка d
    return np.delete(a, d)


def create_by_index(old, mask):  # новый массив по индексам из старого __ 1D
    b = np.array([], float)
    for ind in mask:
        b = np.append(b, [old[ind]])
    return b


def rotation(x_old, y_old, angle):
    x_new = x_old * math.cos(angle) + y_old * math.sin(angle)
    y_new = y_old * math.cos(angle) - x_old * math.sin(angle)
    return [x_new, y_new]


map_xy = np.ndarray(shape=(0, 2), dtype=float)
robot_xy = np.ndarray(shape=(0, 2), dtype=float)
lidar_xy = np.ndarray(shape=(0, 2), dtype=float)
inf_xy = np.ndarray(shape=(0, 2), dtype=float)
interfer_xy = np.ndarray(shape=(0, 2), dtype=float)
laser_xy = np.ndarray(shape=(0, 2), dtype=float)

for i in range(df.shape[0]):
    rads_lidar = np.linspace(-lidar_angle / 360 * np.pi,
                             lidar_angle / 360 * np.pi,
                             len(df.iloc[i]['Lidar_Data']))
    x_l = df.iloc[i]['X'] + math.cos(df.iloc[i]['W']) * 0.3
    y_l = df.iloc[i]['Y'] + math.sin(df.iloc[i]['W']) * 0.3
    robot_xy = np.append(robot_xy, [[df.iloc[i]['X'], df.iloc[i]['Y']]],
                         axis=0)
    lidar_xy = np.append(lidar_xy, [[x_l, y_l]], axis=0)

    x = np.multiply(df.iloc[i]['Lidar_Data'],
                    np.cos(df.iloc[i]['W'] - rads_lidar)) + x_l
    y = np.multiply(df.iloc[i]['Lidar_Data'],
                    np.sin(df.iloc[i]['W'] - rads_lidar)) + y_l

    # x, y = np.multiply(x, np.cos(rot_angle)) + np.multiply(y, np.sin(
    #     rot_angle)), np.multiply(y, np.cos(rot_angle)) - np.multiply(x, np.sin(
    #     rot_angle)) # поворот - НЕ ВКЛЮЧАТЬ
    for k1 in range(len(x)):
        laser_xy = np.append(laser_xy, [[x[k1], y[k1]]], axis=0)

    false_obstacles_indexes = []
    Interferences = []
    for j, el in enumerate(df.iloc[i]['Lidar_Data']):
        if el >= max_range:  # or el <= 0.02
            false_obstacles_indexes.append(j)
        elif 1 and (
                j < threshold * len(df.iloc[i]['Lidar_Data']) or j > (
                1 - threshold) * len(df.iloc[i]['Lidar_Data'])):
            Interferences.append(j)

    x_inf = create_by_index(x, false_obstacles_indexes)
    y_inf = create_by_index(y, false_obstacles_indexes)
    for k2 in range(len(x_inf)):
        inf_xy = np.append(inf_xy, [[x_inf[k2], y_inf[k2]]],
                           axis=0)

    x_body = create_by_index(x, Interferences)
    y_body = create_by_index(y, Interferences)
    for k3 in range(len(x_body)):
        interfer_xy = np.append(interfer_xy, [[x_body[k3], y_body[k3]]],
                                axis=0)

    x_map = del_inf(x, false_obstacles_indexes + Interferences)
    y_map = del_inf(y, false_obstacles_indexes + Interferences)
    for k4 in range(len(x_map)):
        map_xy = np.append(map_xy, [[x_map[k4], y_map[k4]]], axis=0)

    if enable_frames:
        if not os.path.exists('MAP_{}'.format(data_path[:-4])):
            os.makedirs('MAP_{}'.format(data_path[:-4]))

        # plt.figure('Data from the loop{}'.format(i))
        plt.clf()
        plt.title('Data from the loop{}'.format(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x_map, y_map, '.', color='m', label='Map')
        plt.plot(x_inf, y_inf, '.', color='y', label='INF')
        plt.plot(x_body, y_body, '.', color='chocolate', label='Interferences')
        plt.plot(x, y, ':', color='c', label='Laser')
        plt.plot(x_l, y_l, '.', color='r', label='Lidar')  # лидар
        plt.plot(df.iloc[i]['X'], df.iloc[i]['Y'], '.', color='k',
                 label='Robot')  # робот
        plt.legend()
        plt.xlim(-4, 10)
        plt.ylim(-9, 5.0)
        plt.grid()
        plt.savefig('MAP_{}/{}.png'.format(data_path[:-4], i), dpi=600)
    if (i + 1) % 5 == 0: print('{}% completed'.format(i + 1))

plt.figure('Data from the Memory')
plt.title('According to the "{}"'.format(data_path))
plt.xlabel('x')
plt.ylabel('y')
plt.plot(map_xy[:, 0], map_xy[:, 1], '.', color='m', label='Map', zorder=10)
plt.plot(inf_xy[:, 0], inf_xy[:, 1], '.', color='y', label='INF')
plt.plot(interfer_xy[:, 0], interfer_xy[:, 1], '.', color='chocolate',
         label='Interferences')
plt.plot(laser_xy[:, 0], laser_xy[:, 1], ':', color='c', label='Laser')
plt.plot(lidar_xy[:, 0], lidar_xy[:, 1], '.', color='r',
         label='Lidar')  # лидар
plt.plot(robot_xy[:, 0], robot_xy[:, 1], '.', color='k',
         label='Robot')  # робот
plt.legend()
# plt.xlim(-2.2, 10)
# plt.ylim(-7.2, 3.2)
plt.xlim(-4, 10)
plt.ylim(-9, 5.0)
plt.grid()
if not os.path.exists('MAP_{}'.format(data_path[:-4])):
    os.makedirs('MAP_{}'.format(data_path[:-4]))
plt.savefig('MAP_{}/ALL.png'.format(data_path[:-4]), dpi=600)

plt.figure('MAP')
plt.xticks([])
plt.yticks([])
plt.plot(map_xy[:, 0], map_xy[:, 1], '.', color='k')
plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
plt.xlim(-4, 10)
plt.ylim(-9, 5.0)
plt.savefig('MAP_{}/Map_RAW.png'.format(data_path[:-4]), dpi=600)

print(map_xy.shape)

plt.show()
