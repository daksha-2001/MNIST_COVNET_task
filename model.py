from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation,BatchNormalization
# We are importing the Adam Optimizer
from tensorflow.keras.optimizers import Adam
# We are importing the learningratescheduler callback
from tensorflow.keras.callbacks import LearningRateScheduler
import logger

class model:
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()

    def model(self,X_test,X_train,Y_test,Y_train):
        model = Sequential()
        model.add(Conv2D(32, 3, activation='relu', input_shape=(28,28,1)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv2D(16, 3, activation='relu',padding="same")) #24
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv2D(8,3, activation='relu')) #24
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv2D(8, 3, activation='relu',padding="same")) #24
        model.add(MaxPooling2D(pool_size=(2, 2))) #12

        model.add(Conv2D(8,3, activation='relu',padding="same")) #24
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv2D(8,3, activation='relu')) #24
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv2D(8, 3, activation='relu',padding="same")) #24
        model.add(MaxPooling2D(pool_size=(2, 2))) #12

        model.add(Conv2D(8,3, activation='relu')) #24
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

        model.add(Conv2D(10, 3))

        model.add(Flatten())
        model.add(Activation('softmax'))


        model.summary()

        def scheduler(epoch, lr):
            return round(0.003 * 1/(1 + 0.419 * epoch), 10)

        #	LearningRate = LearningRate * 1/(1 + decay * epoch) here decay is 0.319 and epoch is 10.

        # here we are compiling our model and using 'categorical_crossentropy' as our loss function and adam as our optimizer with learning rate =0.003 and metrics is accuracy
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        # Here we are traing our model using the data and using batch size of 128,number of epochs are 20 and using verbose=1 for printing out all the results.
        # In the callbacks parameter we are using the LearningRateScheduler which takes two arguments scheduler function which we built earlier to reduce the learning rate in each decay and verbose =1
        model.fit(X_train, Y_train, batch_size=128, epochs=25, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
        model.save("mymodel")