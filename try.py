import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(1)

while(True):
	    # Capture frame-by-frame
	ret, frame = cap.read()


	upperbody_cascade = cv2.CascadeClassifier('/home/shradha/Downloads/opencv-2.4.13/data/haarcascades/haarcascade_upperbody.xml')


	if ret is True:
		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5,5), 0)
		upper_body = upperbody_cascade.detectMultiScale(gray, 1.02, 5)

	#threshold the image
	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
		thresh = cv2.threshold(gray, 50, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)
	#find contours in thresholded image, then grab the largest
		#storage = cv2.cv.CreateMemStorage (1)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,(0,0))
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		c = max(cnts, key=cv2.contourArea)
		# determine the most extreme points along the contour
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		# draw the outline of the object, then draw each of the
		# extreme points, where the left-most is red, right-most
		# is green, top-most is blue, and bottom-most is teal
		cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
		cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
		cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
		cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
		cv2.circle(frame, extBot, 8, (255, 255, 0), -1)
	    # Display the resulting frame
		cv2.imshow('frame',gray)

    	if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()