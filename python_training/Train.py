import numpy as np

array = [1,2,3,4,5]
array2 = [6,7,8,9,4]

ar1 = np.array(array)
ar2 = np.array(array2)

arr = ar1*ar2

print(" new array is: ", arr)

Index_arr = arr[[]]

print(" Element from (1, 3) is: ", Index_arr)
