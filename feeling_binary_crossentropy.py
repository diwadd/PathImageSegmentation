import numpy as np


def bc(t, o):
    return -np.mean(t*np.log(o) + (1-t)*np.log(1-o))


t = np.array([   0,    0,    0,    1,    1,    1,    0,    0,    0,    1])
o = np.array([0.10, 0.10, 0.10, 0.95, 0.90, 0.90, 0.10, 0.10, 0.10, 0.90])

print(bc(t, o))


n = 10000
t = np.random.randint(2, size=n)
o = np.array([0.01 for i in range(n)])

for i in range(n):
    if t[i] == 1:
        o[i] = 0.99

for i in range(int(0.013*n)):
    if t[i] == 1:
        o[i] = 0.001
    elif t[i] == 0:
        o[i] = 0.999
    else:
        pass

print(bc(t, o))
