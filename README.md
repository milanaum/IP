# IP
pip install opencv-python <br>
************************************************************************************************
1.  Develop  a program to display grayscale image using raed and write operation. <br>
import cv2<br>
img=cv2.imread('rose2.jfif',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/173563349-a090516a-01fc-41ea-9bbe-e408be4d60e8.png)<br>

********************************************************************************************************************************************
2. Develop a program to display the image using matplotlib.<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('dog2.jfif')<br>
plt.imshow(img)<br>

OUTPUT: <br>
<matplotlib.image.AxesImage at 0x20d2d800d30><br>
![image](https://user-images.githubusercontent.com/97940333/173562778-db9f04a6-c59c-4abb-9e7d-08c4f6bf7261.png)<br>

************************************************************************************************************************************************
3. Develop a program to perform linear transformation <br>
 i) Rotation <br>
import cv2 <br>
from PIL import Image<br>
img=Image.open('dog2.jfif')<br>
img=img.rotate(360)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940333/173564580-8dfd6a80-7532-4dd4-a2ff-8f51227b6cd3.png)<br>

****************************************************************************************************************************************************
4. Develop a program to convert color string to RGB color value. <br>
from PIL import ImageColor<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img3=ImageColor.getrgb("green")<br>
print(img3)<br>

OUTPUT: <br>
(255, 0, 0)<br>
(255, 255, 0)<br>
(0, 128, 0)<br>
************************************************************************************************************************************************
5.  Develop a program to create image using colors. <br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(0,128,0))<br>
img.show()<br>

OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940333/173563737-782bccff-798e-4781-a884-b7b3e2591934.png)<br>


****************************************************************************************************************************************************

6. Develop a program to display to initialize the image using various colors spaces. <br> 
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('dog2.jfif')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RG)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.show()<br>

OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940333/173562236-8862725d-baff-4b6a-a2c9-3ec8277aad3f.png)<br>
****************************************************************************************************************************************************************
7. Write a program to display a image attributes. <br>
 from PIL import Image<br>
image=Image.open('dog2.jfif')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

OUTPUT: <br>
Filename: dog2.jfif<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (275, 183)<br>
Width: 275<br>
Height: 183<br>

*********************************************************************************************************************************************************
6. import cv2 <br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('rose1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/173813787-e69b5b4e-9c91-49cb-86e7-b2626e4c25e3.png) <br>
![image](https://user-images.githubusercontent.com/97940333/173813889-330e4f2d-566e-44d1-aed3-e2bc81750542.png) <br>
![image](https://user-images.githubusercontent.com/97940333/173813951-6acee729-d4c1-4a6f-a12c-e80c64cbb119.png) <br>

****************************************************************************************************************************
8. #Convert th eoriginal image to grayscale and then t binary.<br>
import cv2<br>
#read the image file<br>
image=cv2.imread('butterfly2.jpg')<br>
cv2.imshow("RGB",image)<br>
cv2.waitKey(0)<br>
#GrayScale<br>
img=cv2.imread('butterfly2.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
#Binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>


OUTPUT:<br>

![image](https://user-images.githubusercontent.com/97940333/174040564-dd0da5ca-58a6-48c8-8389-1207328dd793.png)<br>


![image](https://user-images.githubusercontent.com/97940333/174040698-41c45fab-0eb6-4986-a08e-cf4bd6dc2735.png)<br>


![image](https://user-images.githubusercontent.com/97940333/174040831-35a398fe-0380-4b16-8def-0cbe8f703e47.png) <br>


************************************************************************************************************************************************

9> #Resize the original image.<br>

import cv2<br>
img=cv2.imread('sunflower.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,100))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

OUTPUT:.<br>

original image length width (183, 275, 3)<br>
Resized image length width (100, 150, 3)<br>
​
![image](https://user-images.githubusercontent.com/97940333/174042826-33f1a80e-3a18-400b-bf78-20c68f4e8e7d.png) <br>
![image](https://user-images.githubusercontent.com/97940333/174044426-d7c5f133-e0a4-499f-a526-3b6be3b11e3c.png) <br>


