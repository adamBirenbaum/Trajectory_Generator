import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import yaml

def make_optimizer(optimizer_dict):

    optim_fun = optimizer_dict.pop('Name')
    fun = getattr(tf.keras.optimizers, optim_fun)

    return fun(**optimizer_dict)

def make_losses(losses_dict):
    loss_fun = losses_dict.pop('Name')
    fun = getattr(tf.keras.losses, loss_fun)

    return fun(**losses_dict)



def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


def make_layer(layer_dict):
    layer_fun = layer_dict.pop('Layer')
    if layer_fun == 'RepeatVector':
        return tf.keras.layers.RepeatVector(layer_dict['n_seq'])
    elif layer_fun == 'TimeDistributed':
        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(layer_dict['n_feat']))
    elif layer_fun == 'Input':
        return tf.keras.layers.Input(eval(layer_dict['shape']))

    fun = getattr(tf.keras.layers, layer_fun)

    return fun(**layer_dict)

def make_model(yaml_file, input_shape):
    yaml_file = yaml.safe_load(open(yaml_file, 'r'))

    #learning_rate = yaml_file['Learning_Rate']

    model_layers = []
    total_layer_str = ''
    for i, layer_dict in enumerate(yaml_file['Layer_Architecture']):
        
        name = layer_dict['Layer']
        model_layers.append(make_layer(layer_dict))

        layer_str = ['{}: {}'.format(layer_name, value) for layer_name, value in layer_dict.items()]
        layer_str = 'Layer {:d} - {}\n\t{}'.format(i, name, '\n\t'.join(layer_str))

        total_layer_str += '\n{}'.format(layer_str)
    optimizer = make_optimizer(yaml_file['Optimizer'])
    lr_metric = get_lr_metric(optimizer)
    losses = make_losses(yaml_file['Losses'])
    model_layers = [tf.keras.layers.Input(input_shape)] + model_layers
    model = models.Sequential(model_layers)
    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics=['mse', lr_metric])

    return model

