import cv2
import numpy as np
import imutils

from flask import Flask, render_template , request ,redirect ,url_for

#making camera object


app =Flask(__name__)

@app.route('/')
def index():
	"""main page of the website only has the weppage"""
	
	return render_template("home1.jinja2")



@app.route('/image', methods=['GET', 'POST'])
def get_image():

	cap=cv2.VideoCapture(0)
	while True:

		ret,frame=cap.read()

		cv2.imshow('frame',frame)

		if cv2.waitKey(10)== ord('s'):

			cv2.imwrite("/home/anushka/Documents/VirtualShop/anuska.jpg", frame)
			break

	cap.release()
	cv2.destroyAllWindows()
	return render_template("home.jinja2")

def get_area():
	"""opens the last stored image from the file
	displays it and checks its upper body proportions
	"""
	upperbody_cascade = cv2.CascadeClassifier("/home/anushka/OpenCV/data/haarcascades/haarcascade_upperbody.xml")

	upper_body = upperbody_cascade.detectMultiScale(gray, 1.02, 5)

	#the scale detected must be sent to another function"""

@app.route('/dress', methods=['GET', 'POST'])
def add_dress():
	"""takes values from function get_area() to find the area of interest of the body"""
	#also opens the last stored image from the file
	#adds the image that are given in the website
	# Load two images from the files

	if request.method == 'POST':
		img1 = cv2.imread('anuska.jpg')

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
	return render_template("home.jinja2")


if __name__ == '__main__':
	
	app.run(debug=True)


































































































