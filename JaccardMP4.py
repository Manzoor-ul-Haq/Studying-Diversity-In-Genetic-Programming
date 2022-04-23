import cv2
import numpy as np

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('pruningJaccard.avi', fourcc, 1, (640, 480))
path = r'C:\Users\admin\Documents\Namal\Fall 2021\CSE-491 Final Year Project-1\Studying-Diversity-In-Genetic-Programming\pruningJaccard\Generation_'

for i in range(0,63):
   img = cv2.imread(path + str(i) + '.png')
   video.write(img)

cv2.destroyAllWindows()
video.release()