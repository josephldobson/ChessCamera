import numpy as np

A = np.array([[[10,5]],[[7,6]],[[11,2]]])
B = sorted(A, key=lambda x: x[0][0] )
print(B)