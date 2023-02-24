data_path = '../examp3.txt'

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


x_l, y_l = 0, 0

for i in range(5):
    x_l_prev, y_l_prev = x_l, y_l

    rads_lidar = np.arange(0, (240 / 360 * 2 * np.pi),
                           240 / 360 * 2 * math.pi / (len(
                               df.iloc[i]['Lidar_Data'])))
    x_l = df.iloc[i]['X'] + math.cos(df.iloc[i]['W']) * 0.3
    y_l = df.iloc[i]['Y'] + math.sin(df.iloc[i]['W']) * 0.3

    x = x_l + np.multiply(
        np.cos(rads_lidar - 2 / 3 * math.pi + df.iloc[i]['W']),
        df.iloc[i]['Lidar_Data'])
    y = y_l + np.multiply(
        np.sin(rads_lidar - 2 / 3 * math.pi + df.iloc[i]['W']),
        df.iloc[i]['Lidar_Data'])

    false_obstacles_indexes = []
    for j, el in enumerate(df.iloc[i]['Lidar_Data']):
        if el == 5.6:
            false_obstacles_indexes.append(j)

    x = del_inf(x)
    y = del_inf(y)

    plt.plot(x, y, '.', color='m')
    plt.plot(df.iloc[i]['X'], df.iloc[i]['Y'], '.',
             color='k')  # положение робота
    plt.plot(x_l, y_l, '.', color='r')  # положение лидара

    if i > 0:
        m1 = x_l_prev - df.iloc[i - 1]['X']
        m2 = x_l - df.iloc[i]['X']
        n1 = y_l_prev - df.iloc[i - 1]['Y']
        n2 = y_l - df.iloc[i]['Y']
        print(math.acos((m1 * m2 + n1 * n2) / (
                (m1 ** 2 + n1 ** 2) ** 0.5 * (m2 ** 2 + n2 ** 2) ** 0.5)))

    else:
        print('0 индекс')

    pass

plt.show()
