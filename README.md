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

![image](https://user-images.githubusercontent.com/97940333/174042826-33f1a80e-3a18-400b-bf78-20c68f4e8e7d.png) <br>
![image](https://user-images.githubusercontent.com/97940333/174044426-d7c5f133-e0a4-499f-a526-3b6be3b11e3c.png) <br>
*************************************************************************************************************************
10.Develop the program to change the image to different color space.<br>
import cv2 <br>
img=cv2.imread(""dog3.jpg"")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176153144-297fcec6-bda1-4a30-a331-61ea0c99a5b2.png) <br>
![image](https://user-images.githubusercontent.com/97940333/176153265-42fb969d-d35e-4f1d-a972-6bb969fbdbf4.png) <br>
![image](https://user-images.githubusercontent.com/97940333/176153353-bcb0d5a1-2e08-4d85-829c-e7c29f5ef743.png) <br>
![image](https://user-images.githubusercontent.com/97940333/176153462-952a3f48-78c3-416c-aeef-0085ece0d2ed.png) <br>
![image](https://user-images.githubusercontent.com/97940333/176153506-82a0db82-d8eb-4070-85c7-2d14a8423896.png) <br>

***************************************************************************************************************************
11. Develop a program to readimage using URL .<br>
 from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://images.unsplash.com/photo-1535591273668-578e31182c4f?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1080&fit=max&ixid=eyJhcHBfaWQiOjM2NTI5fQ'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176397641-1f0adb06-c9d5-4264-87f4-e79975add2e3.png)

****************************************************************************************************************************************************
12. Write a program to perform arithmetic operations on image.<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

img1=cv2.imread('fish7.webp')<br>
img2=cv2.imread('fish8.jpg')<br>

fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>

cv2.imwrite('output.jpg',fimg1)<br>
fimg2 = img1 -img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>

cv2.imwrite('output.jpg',fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>

cv2.imwrite('output.jpg',fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg4)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176401744-bf0bc1b8-7efe-4af8-ad97-0f15e5b6dfe1.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176401830-024e6051-9dd2-4cda-82a4-a0f114a82485.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176401924-a1ca06e4-f0ed-4f72-9e16-00bf735f4523.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176401995-a7e61906-6c84-4421-9f33-88caa6b623c3.png)<br>
True<br>
************************************************************************************************************************************************************
13.Program to createan image using 2D array.<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[25,13,0]<br>
array[:,100:]=[120,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176402997-7ed91093-2075-4218-9803-ea6d975a95d9.png)<br>

-1<br>
****************************************************************************************************************************************************************
14.Develop a program to readimage using URL.<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('dog1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176405174-e07a3bed-cd37-4f0e-a6a3-d70b4ab065cb.png)<br>

hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176405684-28ec3e94-d295-4ae4-95e2-64e3161f1ee6.png)<br>

light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176405847-8f674015-e838-4206-b8d6-a69830251c32.png)<br>

final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_mask)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176406093-4d78b5f6-690d-4ac6-809b-5b087bd56bbc.png)<br>

blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176406261-ffd731ca-397f-4b50-ab5e-840a24c42b8d.png)<br>

************************************************************************************************************************************
15. Bitwise <br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('img1.jpg')<br>
image2=cv2.imread('img1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176410459-795df57b-6379-41fb-b4e3-91ed883496e9.png)<br>
************************************************************************************************************************************************
16. Blurring image.<br>
17. import cv2<br>
import numpy as np<br>

image=cv2.imread('img6.jpg')<br>

cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>

Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>

median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>

bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176420868-96440d4a-1faf-4643-9d60-dc38bc00f3f2.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176420926-af0d7e80-d4f4-4384-981c-3bf8e4f1a81a.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176420980-df09f91c-2eb9-4a5d-a661-3d4c3a83f3b5.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176421074-fa04d3f5-a819-4d24-b0da-2ff562241a92.png)<br>

********************************************************************************************************************************
17.Enhancement<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('min1.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/176424439-ae90310f-5d41-4dcd-9a8e-5f1ef4e2cfd9.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176424614-71f52448-33dc-4d33-9473-b2b57f05ed96.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176424705-b3c117d5-6faf-4ddf-b631-7192859ac230.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176424796-10d08167-5654-4faf-be4d-7122c666459b.png)<br>
![image](https://user-images.githubusercontent.com/97940333/176425102-b61f553a-10ee-401c-9af9-6fa48056f091.png)

****************************************************************************************************************************
18.Morphological.<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('img6.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening= cv2.morphologyEx(img, cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178693582-d612443d-9b81-4d75-b386-c15cf415e573.png)<br>
***************************************************************************************************************************************
19.Grayscale.<br>
import cv2<br>
OriginalImg=cv2.imread('flwr1.jpg')<br>
GrayImg=cv2.imread('flwr1.jpg',0)<br>
isSaved=cv2.imwrite('D:/i.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('the image is successfully saved.')<br>
    
  OUTPUT:<br>
 the image is successfully saved.<br>
![image](https://user-images.githubusercontent.com/97940333/178697391-7b1c3cd7-5012-44ab-ab31-cd903b224d83.png)<br>
![image](https://user-images.githubusercontent.com/97940333/178697488-34317088-7664-45bf-84ad-9b74f3aebb38.png)<br>
****************************************************************************************************************************************************
20.Graylevel slicing with background.<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('moon1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<50):<br>
            z[i][j]=255<br>
        else:<br>
                z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178702917-2689d1ac-df4d-4508-b69e-c2b35d70e95b.png) <br>
****************************************************************************************************************************************************************
21.Graylevel slicing w/o background. <br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('moon1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<50):<br>
            z[i][j]=255<br>
        else:<br>
                z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178706029-f2a6cfaf-bfe4-4211-8194-bd4bae5ced4a.png)
<br>
********************************************************************************************************************************************************************
22.Analyse the image data using histogram with cv2.<br>
import cv2<br>
from matplotlib import pyplot as plt<br>
img = cv2.imread('img1.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256])<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178958570-eb98db4e-ab51-443d-b6dc-30d36dcec35e.png) <br>

#importing required libraries of opencv <br>
import cv2 <br>
 
#importing library for plotting <br>
from matplotlib import pyplot as plt <br>
 
#reads an input image <br>
img = cv2.imread('nature.jpg',0) <br>
 
#find frequency of pixels in range 0-255 <br>
histr = cv2.calcHist([img],[0],None,[256],[0,256]) <br>
 
#show the plotting graph of an image <br>
plt.plot(histr) <br>
plt.show() <br>

OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940333/178959045-93bd020f-f7bd-4b11-82b9-e36af8399302.png)<br>
************************************************************************************************************************************************************
23.Analyse the image data using histogram with numpy.<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('img12.jpg')<br>
hist=cv2.calcHist([img],[0],None,[256],[0,256])<br>
plt.hist(img.ravel(),256,[0,256])<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178959565-c17bb136-3055-4cf6-ae47-ee6cba31592b.png)<br>

import numpy as np<br>
import cv2 as cv<br>
from matplotlib import pyplot as plt<br>
img = cv.imread('img12.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('img12.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br> 
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178959977-13d440d8-9a3b-4cbe-a41a-e3dc302b99f2.png)<br>
![image](https://user-images.githubusercontent.com/97940333/178960117-06236957-5e5e-4729-8520-725b09db23c1.png)<br>

************************************************************************************************************************************************************
23.Analyse the image data using histogram with skimage.<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = cv.imread('img5.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
ax = plt.hist(img.ravel(), bins = 256)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178960622-2867a03a-64c0-4d76-9f6e-f61feadac3c5.png)<br>
![image](https://user-images.githubusercontent.com/97940333/178960930-7ec5b17b-5473-4782-b5c0-82628615fac7.png)<br>

from skimage import io<br>
import matplotlib.pyplot as plt<br>
img = io.imread('lion2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
image = io.imread('lion2.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
plt.show()<br>

OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940333/178961190-2b3a6e4c-6339-403b-8f64-7c497505be7a.png)<br>
![image](https://user-images.githubusercontent.com/97940333/178961285-8cb18da5-ff95-4573-b80e-74921f5c60e8.png)<br>



****************************************************************************************************************************************************
22.Program to perform basic image data analysis using intensity transformation: <br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>

a) Image negative <br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('img2.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/179941838-0fa142ee-2538-496f-beee-dc3788680884.png) <br>

negative =255- pic # neg = (L-1) - img <br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/179942981-24804beb-fa70-4062-97d2-217e106bd1e8.png)<br>


b) LOg transformation <br>
%matplotlib inline <br>

import imageio <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>

pic=imageio.imread('img2.jpg') <br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114]) <br>
gray=gray(pic) <br>

max_=np.max(gray) <br>

def log_transform(): <br>
    return(255/np.log(1+max_))*np.log(1+gray) <br>
plt.figure(figsize=(5,5)) <br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray')) <br>
plt.axis('off'); <br>

OUTPUT:  <br>
![image](https://user-images.githubusercontent.com/97940333/179946062-03dbd291-aa8e-41f0-8652-c17a10457bf4.png) <br>


c) Gamma correction <br>
import imageio <br>
import matplotlib.pyplot as plt<br>

#Gamma encoding<br>
pic=imageio.imread('img2.jpg')<br>
gamma=2.2# Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright<br>

gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/179950018-155ea455-1d91-4c86-850e-aca30e76e365.png) <br>

**********************************************************************************************************************************
24. Program to perform basic image manipulation: <br>
 a) Sharpness <br>
 b) Flipping <br>
 c) Cropping <br>

a) Sharpness <br>
#Image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
# Load the image<br>
my_image = Image.open('sea1.jpg')<br>
#Usw sharpen function<br>
sharp = my_image.filter(ImageFilter.SHARPEN)<br>
#Save the image<br>
sharp.save('D:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/179952626-e8ed9ca4-a737-4da1-8e8c-10a37cb732b5.png) <br>

b) Flipping<br>
#Image flip <br>
import matplotlib.pyplot as plt <br>
#Load the image   <br>
img = Image.open('sea1.jpg') <br>
plt.imshow(img) <br>
plt.show() <br>
#use the flip function <br>
flip = img.transpose(Image.FLIP_LEFT_RIGHT) <br>

#save the image <br>
flip.save('D:/image_flip.jpg') <br>
plt.imshow(flip) <br>
plt.show() <br>

OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940333/179954513-be9898bb-fb97-4df0-9a83-1a7638475eab.png) <br>


c) Cropping <br>
#Importing Image class from PIL module<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
#Opens a image in RGB mode<br>
im = Image.open('sea1.jpg')<br>

#Size of the image in pixels (size of original image)<br>
#(This is not mandatory)<br>
width, height = im.size<br>

#Cropped image of above dimension<br>
#(It will not change original image)<br>
im1 = im.crop((280,100,800,600))<br>

#Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

OUTPUT::<br>
![image](https://user-images.githubusercontent.com/97940333/179958054-610a7ce2-1c47-4502-845b-7ccab11cea54.png) <br>
************************************************************************************************************************************************************
25. Matrix <br>

import numpy as np<br>
import matplotlib.pyplot as plt<br>

arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0, 0, 0)<br>
for y in range(imgsize[1]):<br>
 for x in range(imgsize[0]):<br>
 #Find the distance to the center<br>
  distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>

  #Make it on a scale from 0 to 1innerColor<br>
  distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>

  #Calculate r, g, and b values<br>
  r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
   g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
   b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
   # print r, g, b<br>
   arr[y, x] = (int(r), int(g), int(b))<br>

plt.imshow(arr, cmap='gray')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/180187519-773d378a-0169-4f36-8c5e-232c1623b3e6.png)<br>

************************************************************************************************************************************
from PIL import Image <br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>  
imgsize=(650,650)<br>
image = Image.new('RGBA', imgsize)<br>
innerColor = [153,0,0]<br>
for y in range(imgsize[1]):<br>
for x in range(imgsize[0]):<br>
 distanceToCenter =np.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)<br>
 distanceToCenter = (distanceToCenter) / (np.sqrt(2) * imgsize[0]/2)<br>
 r = distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
 g =  distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
 b =  distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
 image.putpixel((x, y), (int(r), int(g), int(b)))<br>

plt.imshow(image)<br>
plt.show() <br> 

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/180195395-15d02dd3-77cc-45c2-831a-a1bf7485b6f9.png) <br>

**************************************************************************************************************************************
Matrix<br>

import numpy as np<br>
x = np.ones((3, 3))<br>
x[1:-1,1:-1] = 0<br>
x = np.pad(x,pad_width=1,mode='constant',constant_values=2)<br>
print(x)<br>

OUTPUT:<br>

[[2. 2. 2. 2. 2.]
 [2. 1. 1. 1. 2.]
 [2. 1. 0. 1. 2.]
 [2. 1. 1. 1. 2.]
 [2. 2. 2. 2. 2.]] <br>
 
 *************************************************************************************************************************************************
 
 from PIL import Image <br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
w, h = 512, 512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:120, 0:512] = [255, 255, 255]<br>
data[120:256, 0:512] = [218, 218, 218]<br>
data[256:320, 0:512] = [0, 0,0]<br>
data[320:420, 0:512] = [218, 218,218]<br>
data[420:512, 0:512] = [255, 255,255]<br>
# red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('img8.jpg')<br>
img.show()<br>
plt.imshow(img)<br>

OUTPUT: <br>

![image](https://user-images.githubusercontent.com/97940333/181220842-4273d644-f372-4cbd-acac-1f22abed6755.png)<br>

****************************************************************************************************************************************************
 MAXIMUM <br>
 
 import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('img4.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>

print(max_channels)<br>

OUTPUT: <br>

![image](https://user-images.githubusercontent.com/97940333/181224477-028c034b-4778-4cde-b5e6-79c4ae17181b.png) <br>

************************************************************************************************************
MINIMUM <br>

import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('img1.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>

print(min_channels)<br>

OUTPUT:<br>

![image](https://user-images.githubusercontent.com/97940333/181224802-52f64f17-37c4-44ff-8c41-c42c75ab6740.png) <br>

****************************************************************************************************************
AVERAGE <br>

import cv2 <br>
import matplotlib.pyplot as plt <br>
img=cv2.imread("bike1.jpg",0) <br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) <br>
plt.imshow(img) <br>
np.average(img) <br>

OUTPUT: <br>
![image](https://user-images.githubusercontent.com/97940333/181225207-c2a5a4f8-2e7e-4673-a59e-809811c5e6e0.png) <br>
*********************************************************************************************************************
STANDARD DEVIATION <br>

from PIL import Image,ImageStat <br>
import matplotlib.pyplot as plt <br>
im=Image.open('nature.jpg') <br>
plt.imshow(im) <br>
plt.show() <br>
stat=ImageStat.Stat(im) <br>
print(stat.stddev) <br>

OUTPUT: <br>

![image](https://user-images.githubusercontent.com/97940333/181225511-0b025b40-07df-409b-9ffb-420cdb524491.png) <br>
************************************************************************************************************
#Python3 program for printing <br>
#the rectangular pattern <br>
 
#Function to print the pattern <br>
def printPattern(n): <br>
 
arraySize = n * 2 - 1; <br>
result = [[0 for x in range(arraySize)] <br>
for y in range(arraySize)]; <br>
         
#Fill the values <br>
for i in range(arraySize): <br> <br>
for j in range(arraySize): <br>
if(abs(i - (arraySize // 2)) > <br>
abs(j - (arraySize // 2))): <br>
result[i][j] = abs(i - (arraySize // 2)) ; <br>
else: <br>
result[i][j] = abs(j - (arraySize // 2)) ; <br>
             
#Print the array <br>
for i in range(arraySize): <br>
for j in range(arraySize): <br>
print(result[i][j], end = " "); <br>
print(""); <br>
 
#Driver Code <br>
n = 3; <br>
 
printPattern(n); <br>

OUTPUT: <br>

2 2 2 2 2  <br>
2 1 1 1 2  <br>
2 1 0 1 2  <br>
2 1 1 1 2  <br>
2 2 2 2 2  <br>

******************************************************************************************************************************************
import matplotlib.pyplot as plt <br>

M = ([2,2,2,2,2],<br>
     [2,1,1,1,2],<br>
     [2,1,0,1,2],<br>
     [2,1,1,1,2],<br>
     [2,2,2,2,2,])<br>
     
plt.imshow(M)<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/181434227-faf4c971-f081-4c26-9543-3a29cb4a6b06.png) <br>

