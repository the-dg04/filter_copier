import cv2
import numpy as np
from src.filter.filter import Filter

raw_img=cv2.imread('raw_training_image.jpg')
filtered_img=cv2.imread('filtered_training_image.jpg')
myFilter=Filter()
myFilter.train(raw_img,filtered_img)

test_img=cv2.imread('test_image.jpg') # your test image goes here
processed_img=myFilter.apply_filter(test_img)

cv2.imshow("Raw Image",test_img)
cv2.imshow("Filtered Image",processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()