import numpy as np
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)


# 这个函数用于将红色标签图转为白的的标签（其实红色的标签表示灰度值为1(也是只有一个通道）），但不知道为何会显示出红色
def RedToWhite(img_dir, new_img_dir):
    folders = os.listdir(img_dir)  # 得img_dir中所有文件的名字

    for floder in folders:
        image_path = os.path.join(img_dir, floder)
        img = Image.open(image_path)  # 打开图片
        newImg = np.array(img) * 255  # 红色的标签表示灰度值为1,乘以255后都变为255
        newImg = newImg.astype(np.uint8)
        newImg = Image.fromarray(newImg)
        newImg_path = os.path.join(new_img_dir, floder)
        newImg.save(newImg_path)


if __name__ == '__main__':
    img_path = "C:/Users/Administrator/Desktop/masks"
    newImg_path = "C:/Users/Administrator/Desktop/masks-1"
    RedToWhite(img_path, newImg_path)