import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32


classes = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','d1','d2','d3','d4','d6','d9','d12','d14','d15','d16','o1','o2','o3','o4','o5','o6','o7','o8','o10','o11','o12','o13','o14','o15','o16','o17','o18','o19']
class_number = len(classes)

train_accuracy=[]
validation_accuracy=[]
validation_loss=[]
epoc_list=[]

validation_size = 0.2
image_size = 128
channel_number = 3
train_path='train_data'


data = dataset.read_train_sets(train_path, image_size, classes, validation_size=validation_size)


print("successfully read")
print("Training-set:\t\t{}".format(len(data.train.labels)))
print("Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, image_size,image_size,channel_number], name='x')


y_true = tf.placeholder(tf.float32, shape=[None, class_number], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


filter_size_con1 = 3
filter_number_con1 = 32

filter_size_con2 = 3
filter_number_con2 = 32

filter_size_con3 = 3
filter_number_con3 = 64

filter_size_con4 = 3
filter_number_con4 = 64

fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
               input_channel_number, 
               conv_filter_size,        
               filter_number):  
    

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, input_channel_number, filter_number])
    biases = create_biases(filter_number)


    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases


    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    feature_number = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, feature_number])

    return layer


def create_fc_layer(input,          
             input_number,    
             output_number,
             use_relu=True):
    

    weights = create_weights(shape=[input_number, output_number])
    biases = create_biases(output_number)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_con1 = create_convolutional_layer(input=x,
               input_channel_number=channel_number,
               conv_filter_size=filter_size_con1,
               filter_number=filter_number_con1)
layer_con2 = create_convolutional_layer(input=layer_con1,
               input_channel_number=filter_number_con1,
               conv_filter_size=filter_size_con2,
               filter_number=filter_number_con2)

layer_con3= create_convolutional_layer(input=layer_con2,
               input_channel_number=filter_number_con2,
               conv_filter_size=filter_size_con3,
               filter_number=filter_number_con3)

layer_con4= create_convolutional_layer(input=layer_con3,
               input_channel_number=filter_number_con3,
               conv_filter_size=filter_size_con4,
               filter_number=filter_number_con4)
          
flat_layer = create_flatten_layer(layer_con4)

layer_fc1 = create_fc_layer(input=flat_layer,
                     input_number=flat_layer.get_shape()[1:4].num_elements(),
                     output_number=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     input_number=fc_layer_size,
                     output_number=fc_layer_size,
                     use_relu=True) 

					 

layer_fc3 = create_fc_layer(input=layer_fc2,
                     input_number=fc_layer_size,
                     output_number=class_number,
                     use_relu=False) 
					 
					 
y_pred = tf.nn.softmax(layer_fc3,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
	
	acc = session.run(accuracy, feed_dict=feed_dict_train)
	val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
	msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
	print (msg.format (epoch + 1, acc, val_acc, val_loss) )

total_iterations = 0

saver = tf.train.Saver()
def train(iteration_number):
    global total_iterations
    print ('Hello: ' +os.getcwd())
    for i in range(total_iterations,
                   total_iterations + iteration_number):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, os.getcwd()+'\\signature')
    total_iterations += iteration_number
train(iteration_number=1000)




