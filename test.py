import cv2
import numpy as np
from src.filter.filter import Filter
import os

image_directory=os.path.join(os.getcwd(),'public')

raw_img=cv2.imread(os.path.join(image_directory,'raw_training_image.jpg'))
filtered_img=cv2.imread(os.path.join(image_directory,'filtered_training_image.jpg'))
myFilter=Filter()
myFilter.train(raw_img,filtered_img)

test_img=cv2.imread(os.path.join(image_directory,'test_image.jpg')) # your test image goes here
processed_img=myFilter.apply_filter(test_img)

cv2.imshow("Raw Image",test_img)
cv2.imshow("Filtered Image",processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()