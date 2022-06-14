# IP
pip install opencv-python

1.import cv2
img=cv2.imread('rose2.jfif',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/97940333/173563349-a090516a-01fc-41ea-9bbe-e408be4d60e8.png)



2.import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('dog2.jfif')
plt.imshow(img)

<matplotlib.image.AxesImage at 0x20d2d800d30>
![image](https://user-images.githubusercontent.com/97940333/173562778-db9f04a6-c59c-4abb-9e7d-08c4f6bf7261.png)


3.import cv2 
from PIL import image
img=image open('dog2.jfif')
img=img.rotate(360)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/97940333/173564580-8dfd6a80-7532-4dd4-a2ff-8f51227b6cd3.png)


4.from PIL import ImageColor
img2=ImageColor.getrgb("red")
print(img2)
img1=ImageColor.getrgb("yellow")
print(img1)
img3=ImageColor.getrgb("green")
print(img3)

(255, 0, 0)
(255, 255, 0)
(0, 128, 0)

5.from PIL import Image
img=Image.new('RGB',(200,400),(0,128,0))
img.show()

![image](https://user-images.githubusercontent.com/97940333/173563737-782bccff-798e-4781-a884-b7b3e2591934.png)




6.import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('dog2.jfif')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RG)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.show()

![image](https://user-images.githubusercontent.com/97940333/173562236-8862725d-baff-4b6a-a2c9-3ec8277aad3f.png)


7.from PIL import Image
image=Image.open('dog2.jfif')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("Size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()

Filename: dog2.jfif
Format: JPEG
Mode: RGB
Size: (275, 183)
Width: 275
Height: 183






