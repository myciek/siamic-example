import csv

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model


def get_batch(batch_size=64):
    temp_x = x_train
    temp_cat_list = cat_train
    start = 0
    end = train_size
    batch_x = []

    batch_y = np.zeros(batch_size)
    batch_y[int(batch_size / 2):] = 1
    np.random.shuffle(batch_y)

    class_list = np.random.randint(start, end, batch_size)
    batch_x.append(np.zeros((batch_size, 2)))
    batch_x.append(np.zeros((batch_size, 2)))
    print(temp_x.shape)
    print(temp_cat_list)
    print(class_list[0])
    for i in range(0, batch_size):
        batch_x[0][i] = temp_x[np.random.choice(class_list)]
        # If train_y has 0 pick from the same class, else pick from any other class
        if batch_y[i] == 0:
            batch_x[1][i] = temp_x[np.random.choice(class_list)]

        else:
            temp_list = np.append(temp_cat_list[:class_list[i]].flatten(), temp_cat_list[class_list[i] + 1:].flatten())
            batch_x[1][i] = temp_x[np.random.choice(temp_list)]

    return batch_x, batch_y


def nway_one_shot(model, n_way, n_val):
    temp_x = x_val
    temp_cat_list = cat_test
    batch_x = []
    x_0_choice = []
    n_correct = 0

    class_list = np.random.randint(train_size + 1, len(x) - 1, n_val)

    for i in class_list:
        j = np.random.choice(cat_list[i])
        temp = [np.zeros((n_way, 100, 100, 3)), np.zeros((n_way, 100, 100, 3))]
        for k in range(0, n_way):
            temp[0][k] = x[j]

            if k == 0:
                # print(i, k, j, np.random.choice(cat_list[i]))
                temp[1][k] = x[np.random.choice(cat_list[i])]
            else:
                # print(i, k, j, np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i+1:].flatten())))
                temp[1][k] = x[np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i + 1:].flatten()))]

        result = siamese_net.predict(temp)
        result = result.flatten().tolist()
        result_index = result.index(min(result))
        if result_index == 0:
            n_correct = n_correct + 1
    print(n_correct, "correctly classified among", n_val)
    accuracy = (n_correct * 100) / n_val
    return accuracy


light_dir = r'dataset/light_preprocessed'
sound_dir = r'dataset/sound_preprocessed'
train_test_split = 0.7
no_of_items_in_each_class = 100

# Read all the folders in the directory


# Declare training array
cat_list = []
x = []
y = []
y_label = 0
temp = []
# Using just 5 images per category
for file_name in os.listdir(light_dir):
    temp = []
    with open(os.path.join(light_dir, file_name)) as light_file:
        with open(os.path.join(sound_dir, file_name)) as sound_file:
            sound_reader = csv.reader(sound_file)
            light_reader = csv.reader(light_file)
            for row in light_reader:
                temp.append(len(x))
                sound = sound_reader.__next__()[0]
                x.append(np.asarray([row[0], sound]))
                y.append(y_label)
    y_label += 1
    cat_list.append(temp)

cat_list = np.asarray(cat_list)
x = np.asarray(x)
y = np.asarray(y)
print('X, Y shape', x.shape, y.shape, cat_list.shape)
train_size = int(len(x) * train_test_split)
test_size = len(x) - train_size
print(train_size, 'classes for training and', test_size, ' classes for testing')

# Training Split
x_train = x[:train_size]
y_train = y[:train_size]
cat_train = cat_list[:int(train_size / 100)]

# Validation Split
x_val = x[train_size:]
y_val = y[train_size:]
cat_test = cat_list[int(train_size / 100):]

print('X&Y shape of training data :', x_train.shape, 'and', y_train.shape, cat_train.shape)
print('X&Y shape of testing data :', x_val.shape, 'and', y_val.shape, cat_test.shape)

input_shape = (2,)
left_input = Input(input_shape)
right_input = Input(input_shape)

W_init = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2)
b_init = keras.initializers.RandomNormal(mean=0.5, stddev=1e-2)

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_initializer=W_init,
                        bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (7, 7), activation='relu', kernel_initializer=W_init, bias_initializer=b_init,
                        kernel_regularizer=l2(2e-4)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (4, 4), activation='relu', kernel_initializer=W_init, bias_initializer=b_init,
                        kernel_regularizer=l2(2e-4)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init, bias_initializer=b_init,
                        kernel_regularizer=l2(2e-4)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='sigmoid', kernel_initializer=W_init, bias_initializer=b_init)
])

encoded_l = model(left_input)
encoded_r = model(right_input)

subtracted = keras.layers.Subtract()([encoded_l, encoded_r])
prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(subtracted)
siamese_net = Model([left_input, right_input], prediction)

optimizer = Adam(learning_rate=0.0006)
siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer)

plot_model(siamese_net, show_shapes=True, show_layer_names=True)

epochs = 10
n_way = 20
n_val = 100
batch_size = 64

loss_list = []
accuracy_list = []
for epoch in range(1, epochs):
    batch_x, batch_y = get_batch(batch_size)
    loss = siamese_net.train_on_batch(batch_x, batch_y)
    loss_list.append((epoch, loss))
    print('Epoch:', epoch, ', Loss:', loss)
    if epoch % 250 == 0:
        print("=============================================")
        accuracy = nway_one_shot(model, n_way, n_val)
        accuracy_list.append((epoch, accuracy))
        print('Accuracy as of', epoch, 'epochs:', accuracy)
        print("=============================================")
        if (accuracy > 99):
            print("Achieved more than 90% Accuracy")
            # break
