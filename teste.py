import numpy as np
#            vec![21.0, 40.0, 25.0],vec![20.0, 24.0, 11.0],vec![19.0, 17.0, 20.0],
matrix =  np.array([[21.0, 40.0, 25.0],[20.0, 24.0, 11.0],[19.0, 17.0, 20.0]])

# vec![20.0, 24.0, 11.0], vec![19.0, 17.0, 20.0],vec![21.0, 40.0, 25.0],
matrix2 = np.array([[20.0, 24.0, 11.0],[19.0, 17.0, 20.0],[21.0, 40.0, 25.0]])


#element by element multiplication
result = matrix * matrix2

print(result)