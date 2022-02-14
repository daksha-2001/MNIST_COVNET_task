from PIL import Image
import numpy as np
import cv2
from tensorflow import keras
import logger

class predict_class:
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()

    def img_process(self):
        size = 28,28
        image1="custom.png"
        im = Image.open(image1)
        im_resized = im.resize(size)
        im_resized.save("down.png","PNG")
            
        img = cv2.imread("down.png",0)  
        img = img / 255 
        img = np.reshape(img,(1, 28, 28, 1))
        self.log_writer.log(self.log_path,"Prediction_shape:"+str(img.shape))
        return img 
    
    