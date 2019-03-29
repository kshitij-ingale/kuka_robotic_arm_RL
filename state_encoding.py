"""This file implements functions to encode the states. """

import numpy as np

def extend_feature_vec_observation(feature_vec_observation, num_classes=5):
    """Extend sparse target object label to one hot.

    Args:
        feature_vec_observation: list, vec_observation from env.
        num_classes: int, total number of objects.

    Return: list, feature_vec_observation with one hot labels.
    """
    object_label = feature_vec_observation[-1]
    assert object_label < num_classes, 'Object label outside num_classes'
    one_hot_label = np.zeros(num_classes, dtype=np.float32)
    one_hot_label[object_label] = 1
    return feature_vec_observation[:-1] + one_hot_label.tolist()

def test_extend_feature_vec_observation():
    feature_vec_observation = [1,2,3,4,5,6,7,8,9,10,4]
    num_classes = 5
    expected = [1,2,3,4,5,6,7,8,9,10,0,0,0,0,1]
    assert np.isclose(extend_feature_vec_observation(feature_vec_observation,
        num_classes), expected).all()

def get_state_encoding_fn(encoding_method):
    encoding_method_dict = {
        'one_hot_class_extension': extend_feature_vec_observation
    }

if __name__ == '__main__':
    test_extend_feature_vec_observation()
