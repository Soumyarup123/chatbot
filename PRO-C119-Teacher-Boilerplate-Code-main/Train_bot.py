import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from data_preprocessing import preprocess_train_data

def train_bot_model(train_x, train_y):
    model=Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))


    #compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit and save the model
    history= model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    model.save('chatbot_model.h5', history)
    print('model file created and saved')

# Calling Methods to Train Model
train_x, train_y = preprocess_train_data()


train_bot_model(train_x, train_y)