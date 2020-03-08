"""DenseNet models for Keras.
# Reference paper
- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
# Reference implementation
- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras  import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras import models
from tensorflow.keras import utils
import os
# from . import get_submodules_from_kwargs
# from . import imagenet_utils
# from .imagenet_utils import decode_predictions
# from .imagenet_utils import _obtain_input_shape

# BASE_WEIGTHS_PATH = (
#     'https://github.com/keras-team/keras-applications/'
#     'releases/download/densenet/')
# DENSENET121_WEIGHT_PATH = (
#     BASE_WEIGTHS_PATH +
#     'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
# DENSENET121_WEIGHT_PATH_NO_TOP = (
#     BASE_WEIGTHS_PATH +
#     'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
# DENSENET169_WEIGHT_PATH = (
#     BASE_WEIGTHS_PATH +
#     'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
# DENSENET169_WEIGHT_PATH_NO_TOP = (
#     BASE_WEIGTHS_PATH +
#     'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
# DENSENET201_WEIGHT_PATH = (
#     BASE_WEIGTHS_PATH +
#     'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
# DENSENET201_WEIGHT_PATH_NO_TOP = (
#     BASE_WEIGTHS_PATH +
#     'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')

# backend = None
# layers = None
# models = None
# utils = None

WEIGHT_DECAY = 1e-5

def dense_block(x, blocks, name, training=None):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1), training=training)
    return x


def transition_block(x, reduction, name, training=None):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x, training=training)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name, training=None):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x, training=training)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1, training=training)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             training=None):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
    #                      ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, 
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x, training=training)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)
    r = 0.5
    x = dense_block(x, blocks[0], name='conv2', training=training)
    x = transition_block(x, r, name='pool2', training=training)
    x = dense_block(x, blocks[1], name='conv3', training=training)
    if len(blocks) > 2:
        x = transition_block(x, r, name='pool3', training=training)
        x = dense_block(x, blocks[2], name='conv4', training=training)
        x = transition_block(x, r, name='pool4', training=training)
        x = dense_block(x, blocks[3], name='conv5', training=training)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x, training=training)
    x = layers.Activation('relu', name='relu')(x)

    #if include_top:
    x_feat = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', 
      kernel_regularizer=regularizers.l2(WEIGHT_DECAY), name='fc1000')(x_feat)
    # else:
    #     if pooling == 'avg':
    #         x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #     elif pooling == 'max':
    #         x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = models.Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = models.Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = models.Model(inputs, x, name='densenet201')
    else:
        model = models.Model(inputs, x, name='densenet')

    # # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         if blocks == [6, 12, 24, 16]:
    #             weights_path = utils.get_file(
    #                 'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
    #                 DENSENET121_WEIGHT_PATH,
    #                 cache_subdir='models',
    #                 file_hash='9d60b8095a5708f2dcce2bca79d332c7')
    #         elif blocks == [6, 12, 32, 32]:
    #             weights_path = utils.get_file(
    #                 'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
    #                 DENSENET169_WEIGHT_PATH,
    #                 cache_subdir='models',
    #                 file_hash='d699b8f76981ab1b30698df4c175e90b')
    #         elif blocks == [6, 12, 48, 32]:
    #             weights_path = utils.get_file(
    #                 'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
    #                 DENSENET201_WEIGHT_PATH,
    #                 cache_subdir='models',
    #                 file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
    #     else:
    #         if blocks == [6, 12, 24, 16]:
    #             weights_path = utils.get_file(
    #                 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                 DENSENET121_WEIGHT_PATH_NO_TOP,
    #                 cache_subdir='models',
    #                 file_hash='30ee3e1110167f948a6b9946edeeb738')
    #         elif blocks == [6, 12, 32, 32]:
    #             weights_path = utils.get_file(
    #                 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                 DENSENET169_WEIGHT_PATH_NO_TOP,
    #                 cache_subdir='models',
    #                 file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
    #         elif blocks == [6, 12, 48, 32]:
    #             weights_path = utils.get_file(
    #                 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                 DENSENET201_WEIGHT_PATH_NO_TOP,
    #                 cache_subdir='models',
    #                 file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
    #     model.load_weights(weights_path)
    # elif weights is not None:
    #     model.load_weights(weights)

    return model


def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None ,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([1, 1, 1, 1],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([19, 19, 1, 1],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)

def DenseNetS3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([1, 1, 19, 19],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS4(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([10, 10, 10, 10],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS5(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([20, 20],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)

def DenseNetS6(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 12, 8],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS7(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([6, 12, 12, 8],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)

def DenseNetS8(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS9(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 12, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)



def DenseNetS10(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 24, 8],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)



def DenseNetS11(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 24],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)



def DenseNetS12(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([8, 24, 3, 6],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)

def DenseNetS13(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 22],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)

def DenseNetS14(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 20],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)

def DenseNetS15(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 18],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)

def DenseNetS16(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 23],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)



def DenseNetS17(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 25],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)



def DenseNetS18(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 26],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS19(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 27],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS20(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 28],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def DenseNetS20(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                training=None):
    return DenseNet([3, 6, 8, 29],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes, training)


def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format,
                                           mode='torch')

