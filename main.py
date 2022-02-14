# We are importing the Adam Optimizer
from tensorflow.keras.optimizers import Adam
# We are importing the learningratescheduler callback
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import keras
import task 
import logger
import model
import predict
import os

class main:
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()
    def process(self):
        pre_data=task.data(log)
        X_test,X_train,Y_test,Y_train=pre_data.process_data()
        files=os.listdir()
        if "mymodel" in files:
            p=predict.predict_class(log)
            img=p.predict()
            m = keras.models.load_model('mymodel')
            pred=m.predict_classes(img)[0]
            self.log_writer.log(self.log_path,"Prediction is:"+str(pred))
        else:
            model1=model.model(log)
            model_fit=model1.model(X_test,X_train,Y_test,Y_train)
            p=predict.predict(log)
            img=p.predict()
            m = keras.models.load_model('mymodel')
            pred=m.predict_classes(img)[0]
            self.log_writer.log(self.log_path,"Prediction is:"+str(pred))
        

log=open('logg.txt','w+')
main_obj=main(log)
main_obj.process()