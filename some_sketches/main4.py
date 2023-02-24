data_path = '../examp6.txt'

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['X', 'Y', 'W', 'Lidar_Data'])

f = open(data_path)
for line in f:
    readed_line = line
    curr_data = [readed_line[:-1].split(';')[0].split(','),
                 readed_line[:-1].split(';')[1].split(
                     ',')]  # [0] - odo, [1] - lidar
    # print(type(curr_data[0]),len(curr_data), curr_data)
    # print(curr_data[0][0])
    parsed_data = []
    for i, l in enumerate(curr_data):
        # print(l,i)
        if i == 0:
            for j, el in enumerate(l):
                parsed_data.append(float(el))
        else:
            parsed_data.append([float(item) for item in l])
            # print(len(l))
    # в итоге 4 ячейки: float, float, float, list
    # print(parsed_data)
    df.loc[len(df)] = parsed_data

# смотрим DataFrame
df.reset_index(inplace=True)  # доабвляем столбец индексво для подписе точек
print(df)
print(df[['X', 'Y', 'W']])
print(df[['Lidar_Data']])

# 681 точка лидара
# print(df.iloc[0]['Lidar_Data'])

plt.figure('Odometry1')
# plt.subplot(1, 2, 1)
plt.title('Path')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(df['X'], df['Y'], 'o')
ax1 = plt.gca()
df.apply(lambda x: ax1.annotate(x['index'], (x['X'] + 0.0, x['Y'])), axis=1)
plt.legend()
plt.grid()

# plt.subplot(1, 2, 2)
plt.figure('Odometry2')
plt.title('Angle')
plt.xlabel('Count')
plt.ylabel('W')
plt.plot(df['W'], 'o', color='g')
ax2 = plt.gca()
df.apply(lambda x: ax2.annotate(x['index'], (x['index'] + 0.0, x['W'])),
         axis=1)
plt.legend()
plt.grid()

rads = np.arange(0, (240 / 360 * 2 * np.pi), 240 / 360 * 2 * math.pi / 681)
plt.figure('Lidar_0')
ax = plt.subplot(111, polar=True)
ax.plot(rads, df.iloc[0]['Lidar_Data'], 'r')

rads = np.arange(0, (240 / 360 * 2 * np.pi), 240 / 360 * 2 * math.pi / 681)
plt.figure('Lidar_3')
ax = plt.subplot(111, polar=True)
ax.plot(rads, df.iloc[3]['Lidar_Data'], 'r')

plt.figure('MAP')
plt.title('Map')
plt.xlabel('x')
plt.ylabel('y')


def del_inf(a):
    return np.delete(a, false_obstacles_indexes)


for i in range(100):

    rads_lidar = np.arange(-120 / 360 * 2 * np.pi, 120 / 360 * 2 * np.pi,
                           240 / 360 * 2 * math.pi / (len(
                               df.iloc[i]['Lidar_Data'])))
    x_l = df.iloc[i]['X'] + math.cos(df.iloc[i]['W']) * 0.3
    y_l = df.iloc[i]['Y'] + math.sin(df.iloc[i]['W']) * 0.3

    x = np.multiply(df.iloc[i]['Lidar_Data'],
                    np.cos(df.iloc[i]['W'] - rads_lidar)) + x_l
    y = np.multiply(df.iloc[i]['Lidar_Data'],
                    np.sin(df.iloc[i]['W'] - rads_lidar)) + y_l

    x_m = np.multiply(df.iloc[i]['Lidar_Data'], np.cos(rads_lidar)) + x_l + 10
    y_m = np.multiply(df.iloc[i]['Lidar_Data'], np.sin(rads_lidar)) + y_l

    false_obstacles_indexes = []
    for j, el in enumerate(df.iloc[i]['Lidar_Data']):
        if el == 5.6 or el < 0.02:
            false_obstacles_indexes.append(j)

    x = del_inf(x)
    y = del_inf(y)
    x_m = del_inf(x_m)
    y_m = del_inf(y_m)

    plt.plot(x, y, '.', color='m')
    # plt.plot(x_m, y_m, '.', color='k')
    plt.plot(df.iloc[i]['X'], df.iloc[i]['Y'], '.',
             color='k')  # положение робота
    plt.plot(x_l, y_l, '.', color='r')  # положение лидара

    pass

plt.show()
