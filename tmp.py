import numpy as np

x = 100

print("2: {} -> {}".format(np.power(x, 1/2), np.power(np.power(x, 1/2), 2)))
print("3: {} -> {}".format(np.power(x, 1/3), np.power(np.power(x, 1/3), 3)))
print("4: {} -> {}".format(np.power(x, 1/4), np.power(np.power(x, 1/4), 4)))