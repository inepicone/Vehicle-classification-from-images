import tensorflow                  as tf
import keras                       as k

def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """
    
    data_aug_layers = []
    
    for key in data_aug_layer.keys():
        if key == "random_flip":
            rf = k.layers.RandomFlip(**data_aug_layer['random_flip'])
            data_aug_layers.append(rf)
                    
        if key == "random_rotation":
            rr = k.layers.RandomRotation(**data_aug_layer['random_rotation'])
            data_aug_layers.append(rr)
        
        if key == "random_zoom":
            rz = k.layers.RandomZoom(**data_aug_layer['random_zoom'])
            data_aug_layers.append(rz)
        
    data_augmentation = tf.keras.Sequential(data_aug_layers)


    return data_augmentation

