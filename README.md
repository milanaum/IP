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
                z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97940333/178703759-95fd8137-59c8-4b22-80f9-644288a8afce.png) <br>
********************************************************************************************************************************************************************


