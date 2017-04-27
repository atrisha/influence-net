from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

s = [1,2,3,4]
f = [6,7,8,9]

k = [[e1,e2] for e1,e2 in zip(s,f)]
g = np.asarray(k)
print(k)