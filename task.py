
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
# We are importing the Adam Optimizer
from tensorflow.keras.optimizers import Adam
# We are importing the learningratescheduler callback
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation,BatchNormalization
import logger


class data:
    
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()

    def process_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

        #Doing type conversion or changing the datatype to float32 for the data
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        #Doing standardization or normalization here dividind each pixel by 255 in the train and test data
        X_train /= 255
        X_test /= 255

        # simply we can say we are doing sort of onehot encoding
        Y_train = to_categorical(y_train, 10)
        Y_test = to_categorical(y_test, 10)

        self.log_writer.log(self.log_path,"X_train shape:"+str(X_train.shape))
        self.log_writer.log(self.log_path,"X_test shape:"+str(X_test.shape))
        self.log_writer.log(self.log_path,"Y_train shape:"+str(Y_train.shape))
        self.log_writer.log(self.log_path,"Y_test shape:"+str(Y_test.shape))

        return X_test,X_train,Y_test,Y_train

