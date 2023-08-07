import os
import cv2
DATADIR="/export/home/ra35tiy/vae-inn-synthesis-prober/src/data/ffhq_images/"

IMG_SIZE=128
path=os.path.join(DATADIR) 

img_list=os.listdir(path)



for i in range(len(img_list)):
  print(img_list[i])
  img_array=cv2.imread(os.path.join(path,img_list[i]),cv2.IMREAD_COLOR)
  new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
  img_name=str(i)+'.jpg'
  save_path='/export/home/ra35tiy/vae-inn-synthesis-prober/src/data/ffhq_images_new/'+str(img_name)
  print(save_path)
  cv2.imwrite(save_path,new_array)


