'''
Created on Aug 22, 2018

@author: en
'''
import tensorflow as tf


def train_cambridge_dataset(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=''),
                "pose": tf.FixedLenFeature((), tf.string, default_value=''),
               }
    parsed_features = tf.parse_single_example(example_proto, features)

    feature = tf.decode_raw(parsed_features["image"], tf.float32)
    feature = tf.reshape(feature, (2, 256, 455, 3))
    pose = tf.decode_raw(parsed_features["pose"],  tf.float32)
    pose = tf.reshape(pose, (26, 7))
    
    feature = tf.random_crop(feature, (2, 224, 224, 3))

    return tf.cast(feature, tf.float32), pose[0]# pose[extra_R[0]*4 + extra_R[1]]


def test_cambridge_dataset(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=''),
                "pose": tf.FixedLenFeature((), tf.string, default_value=''),
                "extra": tf.FixedLenFeature((), tf.string, default_value=''),
               }
    parsed_features = tf.parse_single_example(example_proto, features)

    feature = tf.decode_raw(parsed_features["image"], tf.float32)
    feature = tf.reshape(feature, (2, 256, 455, 3))
    pose = tf.decode_raw(parsed_features["pose"],  tf.float32)
    pose = tf.reshape(pose, (26, 7))
    feature = feature[:, 16:240, 116:340, :]
    return tf.cast(feature, tf.float32), pose[0]

def train_cambridge_dataset_extra(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=''),
                "pose": tf.FixedLenFeature((), tf.string, default_value=''),
               }
    parsed_features = tf.parse_single_example(example_proto, features)

    feature = tf.decode_raw(parsed_features["image"], tf.float32)
    feature = tf.reshape(feature, (2, 256, 455, 3))
    pose = tf.decode_raw(parsed_features["pose"],  tf.float32)
    pose = tf.reshape(pose, (26, 7))
    
    feature = tf.random_crop(feature, (2, 224, 224, 3))

    return tf.cast(feature, tf.float32), pose[0], pose[16], pose[20] 

def test_cambridge_dataset_extra(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=''),
                "pose": tf.FixedLenFeature((), tf.string, default_value=''),
                "extra": tf.FixedLenFeature((), tf.string, default_value=''),
               }
    parsed_features = tf.parse_single_example(example_proto, features)

    feature = tf.decode_raw(parsed_features["image"], tf.float32)
    feature = tf.reshape(feature, (2, 256, 455, 3))
    pose = tf.decode_raw(parsed_features["pose"],  tf.float32)
    pose = tf.reshape(pose, (26, 7))
    
    feature = feature[:, 16:240, 116:340, :]
    return tf.cast(feature, tf.float32), pose[0], pose[16], pose[20] 



def get_read_tfrecord_fn(dataset, extra_loss=False):
    print('**********************', dataset.lower(), '*****************************')
    if dataset in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
        if extra_loss:
            return train_cambridge_dataset_extra, test_cambridge_dataset_extra
        else:
            return train_cambridge_dataset, test_cambridge_dataset
    else:
        raise NotImplementedError