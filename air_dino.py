from selenium import webdriver
import pyautogui
import math
import cv2
import numpy as np

driver_path = 'your-chromedriver-path'
wd = webdriver.Chrome(executable_path=driver_path)

# launch chrome in python environment
wd.get('http://www.trex-game.skipser.com/')

webcam = cv2.VideoCapture(0)

while webcam.isOpened():

    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # filter region of interest
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 2)

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # skin color range in HSV
    lower_skin = np.array([0, 48, 80], dtype="uint8")
    upper_skin = np.array([20, 255, 255], dtype="uint8")

    # identify skin color (white will be skin color and rest will be black)
    mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)

    # remove noise
    erosion = cv2.erode(mask, kernel=np.ones((3, 3), dtype=np.uint8), iterations=3)
    dilation = cv2.dilate(erosion, kernel=np.ones((3, 3), dtype=np.uint8), iterations=3)

    blur = cv2.GaussianBlur(dilation, (5, 5), 100)
    ret_1, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    # find contours
    thresh_copy = thresh.copy()
    contours, hierarchy = cv2.findContours(thresh_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # find max contour (the hand, obviously)
        max_contour = max(contours, key=lambda x: cv2.contourArea(x))

        # draw the max contour
        cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 1)

        # Find convex hull and then convexity defects
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        defects_count = 0

        # find angle of the convex defect
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            if angle <= 90:
                defects_count += 1

            cv2.line(roi, start, end, [0, 255, 0], 2)

        if defects_count >= 3:
            cv2.putText(frame, "Jump", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            pyautogui.press('space')
        else:
            pass

    except Exception:
        pass

    cv2.imshow('frame', frame)
    cv2.imshow('thresh', thresh)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
