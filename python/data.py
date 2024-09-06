#!/bin/python

import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) == 1:
    print("Enter bitmap number")
    exit(-1)

img=int(sys.argv[1])

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#print(class_names[test_labels[img]])
#print(': ' + test_labels[img])

sys.stderr.write("Number: " + str(train_labels[img]) + "\n")

print ('{ "id": "req", "inputs": [ { "name": "bitmap", "shape": [1,28,28], "datatype": "FP32", "data": ')
print(np.array2string(train_images[img].flatten()/255,max_line_width=1000000,floatmode="fixed",separator=",",precision=8))
print ('} ], "outputs": [ { "name": "probabilities" } ] }')



