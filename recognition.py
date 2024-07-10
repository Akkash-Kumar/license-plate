import cv2  #opencv

import pytesseract   #text reading from img
  

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'    #filepath where tesseract ocr is installed

image = cv2.imread('car4.jpg')    #read car4.jpg file as ip

##cv2.imshow('car',image)   #display car4 image


grayImg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   #convert to grayscale
 
##cv2.imshow('grayimg',grayImg)     #display grayscale img


canny_edge = cv2.Canny(grayImg,170,200)  #apply Canny edge detection

##cv2.imshow('cannyImg',canny_edge)  #display canny img


contours,new = cv2.findContours(canny_edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   #find contours

contours = sorted(contours, key = cv2.contourArea, reverse = True) [:20]    #sort 30 contours in descending order(big size to small size contours)


contour_with_license_plate = None    

license_plate = None

x=None   #initialise coordinates of license plate 

y=None

w=None

h=None   #end


contour_image = image.copy()   #draw contour in which image,and make copy

cv2.drawContours(contour_image,contours,-1,(0,255,0),2)   #draw contours

##cv2.imshow('contourImg',contour_image)  #display contour img


for contour in contours:

    perimeter = cv2.arcLength(contour,True)   #length of contour

    approx = cv2.approxPolyDP(contour,0.01 * perimeter ,True)  #to get contour in polygon shapes like square, rectangle

    if len(approx) ==4:   #to see whether contour is rectangle

        contour_with_license_plate = approx

        x,y,w,h = cv2.boundingRect(contour)   #to find coordinates

        license_plate = grayImg[y:y+h,x:x+w]  #to crop license plate

        break


(thresh,license_plate) = cv2.threshold(license_plate,127,255,cv2.THRESH_BINARY)  #convert to threshold pic


license_plate = cv2.bilateralFilter(license_plate,11,17,17)   #remove noise


text = pytesseract.image_to_string(license_plate)   #text recognition


image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)  #draw rectangle

image = cv2.putText(image,text,(x-100,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)   #put text above rectangle

cv2.imshow('final',image)  #display op

print("license plate :", text) 
        








