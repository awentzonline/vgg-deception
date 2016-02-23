'''Fool VGG16 with Keras.
Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.
Run the script with:
```
python deception.py path_to_your_base_image.jpg prefix_for_results desired_class
```
e.g.:
```
python deception.py img/tuebingen.jpg results/my_result 7
```
It is preferrable to run this script on GPU, for speed.
If running on CPU, prefer the TensorFlow backend (much faster).

Adapted from other Keras examples.
'''

from __future__ import print_function
from scipy.misc import imread, imresize, imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import os
import argparse
import h5py

from keras.models import Sequential
from keras.objectives import categorical_crossentropy
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

parser = argparse.ArgumentParser(description='Fool VGG16 with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('imagenet_class', metavar='in_class', type=int,
                    help='ImageNet class id.', default=0)
args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix
imagenet_class = args.imagenet_class
weights_path = 'vgg16_weights.h5'


# dimensions of the generated picture.
img_width = 224
img_height = 224

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = imresize(imread(image_path), (img_width, img_height))
    img = img.transpose((2, 0, 1)).astype('float64')
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# get tensor representations of our images
input_image = K.placeholder((1, 3, img_width, img_height))
# build the VGG16 network with our 3 images as input
first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
first_layer.input = input_image
model = Sequential()
model.add(first_layer)
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
model.load_weights(weights_path)
# f = h5py.File(weights_path)
# for k in range(f.attrs['nb_layers']):
#     if k >= len(model.layers):
#         # we don't look at the last (fully-connected) layers in the savefile
#         break
#     g = f['layer_{}'.format(k)]
#     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
#     model.layers[k].set_weights(weights)
# f.close()
# print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.get_output()) for layer in model.layers])

output = model.get_output()
desired_output = np_utils.to_categorical([imagenet_class], 1000)
loss = categorical_crossentropy(desired_output, output)
# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, input_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([input_image], outputs)
f_class_output = K.function([input_image], output)

def eval_loss_and_grads(x):
    x = x.reshape((1, 3, img_width, img_height))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to maximize the activation of our selected class.
try:
    x = preprocess_image(base_image_path)
    predicted_classes = f_class_output([x])[0]
    print('Initial class prediction {}'.format(np.argmax(predicted_classes)))
    for i in range(10):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        end_time = time.time()
        print('Current loss value:', min_val)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
except KeyboardInterrupt:
    pass

# save current generated image
img = deprocess_image(x.reshape((3, img_width, img_height)))
fname = result_prefix + '_at_iteration_%d.png' % i
imsave(fname, img)
print('Image saved as', fname)

# did we fool the network?
x = x.reshape((1, 3, img_width, img_height))
predicted_classes = f_class_output([x])[0]
print('New class prediction {}'.format(np.argmax(predicted_classes)))
