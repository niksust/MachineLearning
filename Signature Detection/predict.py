import cv2
import os
import glob
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

path_dir = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = path_dir + '/' + image_path + '/'
path = os.path.join('test_data', filename, '*g')
files = glob.glob(path)


classes = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','d1','d2','d3','d4','d6','d9','d12','d14','d15','d16','o1','o2','o3','o4','o5','o6','o7','o8','o10','o11','o12','o13','o14','o15','o16','o17','o18','o19']
class_number = len(classes)
image_size = 128
channel_number = 3

totalExperimented =0;
truePositive=0;

for i in files:

    filename = i

    p = 0
    className = ""
    chk2=0;
    i = i[::-1]
    for kk in i:
        if kk == '.':
            chk2=1;
        elif kk == '\\':
            break;
        elif kk == '/':
            break;
        elif chk2 == 1:
            className = className + kk;


    className = className[::-1]

    images = []
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    x_batch = images.reshape(1, image_size, image_size, channel_number)

    sess = tf.Session()
    saver = tf.train.import_meta_graph('signature.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, class_number))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
 
    j = 1;
    val =0;
    probableClass = 0;
    for ii in result:
        chk=0
        for jj in ii:
            chk=chk+1
            if jj>val:
                val =jj
                probableClass = chk
    totalExperimented = totalExperimented +1
    if val >= 0.90:

        print('Photo Name       : ', className)
        print('predicted person : ', classes[probableClass - 1])
        print('Probability value: ', val)
        truePositive = truePositive + 1
    else:
        print('Photo Name       : ', className)
        print('predicted person : Unknown')
        print('Probability value: ', val)

    print()
    print()
 
print('Total    : ', truePositive)
print('Unknown  : ', totalExperimented-truePositive)
