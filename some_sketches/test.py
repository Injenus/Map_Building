import numpy as np
import math
import cv2
from scipy import ndimage

a = np.array(([1,1,0],
                [1,1,1],
                [1,1,2],
                [1,1,3]))

print(ndimage.center_of_mass(a))
imf=cv2.imread()
