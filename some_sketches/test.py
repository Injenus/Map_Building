import numpy as np

# arrayXY = np.ndarray(shape=(0,2), dtype=int)
# print(arrayXY)
# x=4
# y=6
# arrayXY = np.append(arrayXY, [[x, y]], axis = 0)
# arrayXY = np.append(arrayXY, [[x, y]], axis = 0)
#
# print(arrayXY[:, 1])

q = [1, 4, 5]
o = np.array([1.0, 2.04, 3.0, 4.0, 5., 6., 7.0], float)


def create_by_index(old):  # новый массив по индексам из старого
    b = np.array([],float)
    for ind in q:
        b = np.append(b, [old[ind]])
    return b


w = create_by_index(o)
print(w)
