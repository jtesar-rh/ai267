#!/bin/python

from PIL import Image
import numpy as np
import sys


if len(sys.argv) == 1:
    print("Enter bitmap number")
    exit(-1)


im = Image.open(sys.argv[1])
p = np.array(im)

print ('{ "id": "req", "inputs": [ { "name": "bitmap", "shape": [1,28,28], "datatype": "FP32", "data": ')
print(np.array2string(p.flatten()/255,max_line_width=1000000,floatmode="fixed",separator=",",precision=8))
print ('} ], "outputs": [ { "name": "probabilities" } ] }')



