import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.resize(cv2.imread('component_knock_off/image.jpg', 0),
                 (0, 0), fx=0.8, fy=0.8)
template = cv2.resize(cv2.imread(
    'component_knock_off/template.jpg', 0), (0, 0), fx=0.8, fy=0.8)
h, w = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()

    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(img2, location, bottom_right, 255, 5)
#     plt.imshow(img2, cmap='gray')
    x1, y1 = location
    x2, y2 = bottom_right
    im = img.copy()
    crop_img = img2[int(y1):int(y2), int(x1):int(x2)]
#     crop_img = img2[y1:y2, x1:x2]
    plt.imshow(crop_img, cmap='gray')
# cv2.imwrite('croping1.jpg',crop_img)
