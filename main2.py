import cv2
import imutils
import numpy as np
def image():
	cap=cv2.VideoCapture(0)
	while True:
		ret,frame=cap.read()
		
		cv2.imshow('frame',frame)
		
		if cv2.waitKey(1)== ord('q'):
			
			cv2.imwrite("/home/anushka/Documents/e.jpg", frame)
			break

	cap.release()
	cv2.destroyAllWindows()

def contours():

 
# load the image, convert it to grayscale, and blur it slightly
	image = cv2.imread("e.jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(image, (5, 5), 0)
	 
	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest
	# one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	 
	cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
	cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
	cv2.circle(image, extRight, 8, (0, 255, 0), -1)
	cv2.circle(image, extTop, 8, (255, 0, 0), -1)
	cv2.circle(image, extBot, 8, (255, 255, 0), -1)
	 
	# show the output image
	
	cv2.imwrite("/home/anushka/Documents/f.jpg", image)
	cv2.waitKey(0)

def add():

	img1 = cv2.imread('f.jpg')

	#the dress will be given by the user through the webpage
	img2 = cv2.imread('dress2.jpeg')

	# I want to put logo on top-left corner, So I create a ROI
	rows,cols,channels = img2.shape
	roi = img1[180:rows+180, 180:cols+180] #the region of interest is x=180 to 439 and y=180 to 374  
	#the area of interest will be given by function above
	print('Rows:',rows+180)
	print('cols:', cols+180)
	# Now create a mask of logo and create its inverse mask
	img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	# add a threshold
	ret, mask1 = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

	mask_inv = cv2.bitwise_not(mask1)

	# Now black-out the area of logo in ROI
	img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

	# Take only region of logo from logo image.
	img2_fg = cv2.bitwise_and(img2,img2,mask = mask1)


	dst = cv2.add(img1_bg,img2_fg)
	img1[180:rows+180, 180:cols+180 ] = dst

	cv2.imshow('res',img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__== '__main__':
	image()
	contours()
	add()