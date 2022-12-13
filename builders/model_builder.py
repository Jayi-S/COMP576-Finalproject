from models import *
import tensorflow as tf

layers = tf.keras.layers


def builder(num_classes, input_size=(256, 256), model='UNet', base_model=None):
    models = {
              'UNet': UNet,
              'DeepLabV3': DeepLabV3,
              'DeepLabV3Plus': DeepLabV3Plus}

    assert model in models

    net = models[model](num_classes, model, base_model)

    inputs = layers.Input(shape=input_size+(3,))

    return net(inputs), net.get_base_model()
