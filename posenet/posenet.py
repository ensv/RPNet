'''
Created on Apr 3, 2018

@author: en
'''
import argparse, sys, os
from tensorflow.python.framework.errors_impl import OutOfRangeError
import math
from sklearn.externals import joblib
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from abstract_network.setting import Setting
from abstract_network.abstract import AbstractNetwork
from GoogLeNet.googlenet import GoogLeNet
import tensorflow as tf
import numpy as np


slim = tf.contrib.slim  # @UndefinedVariable


def train_preprocessing_cambridge_dataset(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=''),
                "pose": tf.FixedLenFeature((), tf.string, default_value=''),
               }
    parsed_features = tf.parse_single_example(example_proto, features)

    feature = tf.decode_raw(parsed_features["image"], tf.float32)
    feature = tf.reshape(feature, (256, 455, 3))
    pose = tf.decode_raw(parsed_features["pose"],  tf.float32)

    feature = tf.random_crop(feature, (224, 224, 3))
    return feature, pose


def test_preprocessing_cambridge_dataset(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=''),
                "pose": tf.FixedLenFeature((), tf.string, default_value=''),
               }
    parsed_features = tf.parse_single_example(example_proto, features)

    feature = tf.decode_raw(parsed_features["image"], tf.float32)
    feature = tf.reshape(feature, (256, 455, 3))
    pose = tf.decode_raw(parsed_features["pose"],  tf.float32)

    feature = feature[16:240, 116:340, :]
    return feature, pose


class PoseNet(AbstractNetwork):
    '''
    classdocs
    '''


    def __init__(self, flags):
        '''
        Constructor
        '''

        flags.name = 'PoseNet'
        self.output_dim = 7

        self.wd = flags.wd
        self.beta = flags.beta
        setting = Setting('absolute_pose_network')
        AbstractNetwork.__init__(self, setting, flags)
        self.train_dir = self._train_dir_path('_%d_%.4f'%(self.beta, self.wd))
        if self.train_dir == '': return
        self._get_train_test_set()

        self.prepare_data(train_preprocessing_cambridge_dataset, test_preprocessing_cambridge_dataset)

        self.prepare_inference()
        self.prepare_loss()
        self.get_train_op()
        self.max_iteration = 80000
        
    
    def _get_train_test_set(self):
        
        if self.dataset in self.cambridge_subset:
            path = self.setting.data_path.replace('absolute_pose_network', 'absolute_cambridge') + '/' + self.dataset
            self.train_filenames = '%s/train.tfrecord'%path
            self.test_filenames = '%s/test.tfrecord'%path
            self.valid_filenames = '%s/validation.tfrecord'%path
        else:
            raise NotImplementedError 
        
    
    def before_train(self, sess):
        pass

    def before_valid_test(self, sess):
        pass

    def prepare_inference(self):
        
#         arg_scope = inception.inception_utils.inception_arg_scope(weight_decay=self.wd)
#         with slim.arg_scope(arg_scope):
        net = GoogLeNet({'data': self.input_X})
        
        weight_vars = [v for v in tf.global_variables() if "weights" in v.name]
        print('weight var: ', len(weight_vars))
        
        if self.wd > 0:
            for v in tf.global_variables():
                if "weights" in v.name:
                    tmp_loss = tf.nn.l2_loss(v) * self.wd
                    tf.add_to_collection('losses', tmp_loss)
        
        self.xyz_logits = net.layers['cls3_fc_pose_xyz']
        self.wpqr_logits = net.layers['cls3_fc_pose_wpqr']
        self.aux1_xyz_logits = net.layers['cls1_fc_pose_xyz']
        self.aux1_wpqr_logits = net.layers['cls1_fc_pose_wpqr']
        self.aux2_xyz_logits = net.layers['cls2_fc_pose_xyz']
        self.aux2_wpqr_logits = net.layers['cls2_fc_pose_wpqr']
        
        self.net = net
         
    def prepare_loss(self):
        self.tran_loss = self._l2_norm_loss(self.xyz_logits, labels=self.input_Y[:, :3], weights=1.0, scope='Train/translation_loss')
        self.rotat_loss = self._l2_norm_loss(self.wpqr_logits, labels=self.input_Y[:, 3:], weights=self.beta, scope='Train/rotation_loss')
        
        # Aux classifiers
        self._l2_norm_loss(self.aux2_xyz_logits, labels=self.input_Y[:, :3], weights=.3, scope='Train/aux2_translation_loss')
        self._l2_norm_loss(self.aux2_wpqr_logits, labels=self.input_Y[:, 3:], weights=self.beta/3.33, scope='Train/aux2_rotation_loss')
        self._l2_norm_loss(self.aux1_xyz_logits, labels=self.input_Y[:, :3], weights=.3, scope='Train/aux1_translation_loss')
        self._l2_norm_loss(self.aux1_wpqr_logits, labels=self.input_Y[:, 3:], weights=self.beta/3.33, scope='Train/aux1_rotation_loss')

    def _load_pretrained_weight(self, sess):
        if self.dataset in self.cambridge_subset:
            self.net.load('../GoogLeNet/weights/posenet.npy', sess)
        else:
            raise NotImplementedError
    def _define_additional_summaries(self):
        self.valid_loss, self.valid_error = tf.Variable([0, 0], dtype='float32', trainable=False), tf.Variable([0, 0], dtype='float32', trainable=False)

        self.valid_summary = [tf.summary.scalar('Valid/tran_loss', self.valid_loss[0]), 
                         tf.summary.scalar('Valid/rotat_loss', self.valid_loss[1]),
                         tf.summary.scalar('Valid/tran_error', self.valid_error[0]), 
                         tf.summary.scalar('Valid/rotat_error', self.valid_error[1]),
                         ]  # @UndefinedVariable
    
    def evaluate_validation(self, sess):
        # example of prototype
        
        self.before_valid_test(sess)
        sess.run(self.validation_init_op)

        loss = []
        error = [[], []]
        # train for half epoch
        while True:
            try:
                tmp = sess.run([self.tran_loss, self.rotat_loss, self.xyz_logits, self.wpqr_logits, self.input_Y])                
                loss.append([tmp[0], tmp[1]])
                tmp[3] = tmp[3]/ np.repeat(np.linalg.norm(tmp[3], axis=1).reshape((-1, 1)), 4, axis=1)
                d = np.abs(np.sum(np.multiply(tmp[3], tmp[4][:, 3:]), axis=1))
                np.putmask(d, d> 1, 1)
                error[1].append(2 * np.arccos(d) * 180/math.pi)
                error[0].append(np.linalg.norm(tmp[2]-tmp[4][:, :3], axis=1))
            except OutOfRangeError:
                break
        
        # validation loss
        loss = np.array(loss).mean(axis=0)
        sess.run(self.valid_loss.assign(loss))
        
        # validation error
        error = [np.concatenate(error[0]), np.concatenate(error[1])]
        error = np.median(np.array(error), axis=1)
        sess.run(self.valid_error.assign(error))
        
        summary_str = sess.run(self.valid_summary)
        for each in summary_str: 
            self.summary_writer.add_summary(each, sess.run(tf.train.get_global_step()))
    
    def _custom_evaluation(self, sess):

        loss = []
        error = [[], [], []]
        pose = []
        # train for half epoch
        while True:
            try:
                tmp = sess.run([self.tran_loss, self.rotat_loss, self.xyz_logits, self.wpqr_logits, self.input_Y])

                pose.append(np.hstack((tmp[2], tmp[3], tmp[4])))                
                loss.append([tmp[0], tmp[1]])
                
                tmp[3] = tmp[3]/ np.repeat(np.linalg.norm(tmp[3], axis=1).reshape((-1, 1)), 4, axis=1)
                d = np.abs(np.sum(np.multiply(tmp[3], tmp[4][:, 3:]), axis=1))
                np.putmask(d, d> 1, 1)
                error[1].append(2 * np.arccos(d) * 180/math.pi)
                error[0].append(np.linalg.norm(tmp[2]-tmp[4][:, :3], axis=1))
                
                tmp[4][:, :3] = tmp[4][:, :3] / np.linalg.norm(tmp[4][:, :3], axis=1, keepdims=True)
                tmp[2] = tmp[2] / np.linalg.norm(tmp[2], axis=1, keepdims=True)
                error[2].append(np.arccos(np.sum(tmp[4][:, :3] * tmp[2], axis=1)) * 180/math.pi)
            except OutOfRangeError:
                break

        # all the pose
        pose = np.vstack(pose)
        np.save('%s/all_pose'%self.train_dir, pose)
        
        # validation loss
        loss = np.array(loss).mean(axis=0)
        
        # validation error
        error = [np.concatenate(error[0]), np.concatenate(error[1]), np.concatenate(error[2])]
        error = np.median(np.array(error), axis=1)
        
        joblib.dump(np.concatenate((loss, error)), '%s/%s.pkl'%(self.train_dir, self.test_result_fn), compress=3)
    
def main(_):
    
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if FLAGS.experiment == 'cambridge':
        beta = {'KingsCollege':500, 
                'OldHospital':100, 
                'ShopFacade':100, 
                'StMarysChurch':250, 
                'Street':2000}
                
        FLAGS.experiment = 'cambridge'
        for _ in range(2):
            for dataset in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
                for lr in [1e-5]:
                    for wd in [1e-2]:
                        FLAGS.wd = wd
                        FLAGS.lr = lr
                        FLAGS.dataset = dataset
                        FLAGS.beta = beta[dataset]
                        net = PoseNet(FLAGS)
                        net.process()
    else:
        print ('unknown experimentation')
        raise NotImplementedError
    os.chdir(working_dir)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
    parser.add_argument("--optimization", default= 'sgdm', type=str, help= "Optimization strategy: adadelta, sgd, sgdm")
    parser.add_argument("--lr", type = float, default = 1e-5, help = "Initial learning rate")
    parser.add_argument("--lr_decay_rate", type = float, default = 0.90, help = 'decay rate for learning rate')
    parser.add_argument("--lr_decay_step", type = int, default = 80, help = 'nb of epochs to decay learning rate, use big number of step to avoid decaying the lr')
    parser.add_argument("--nb_epoches", type = int, default = 10000, help = "Nb of epochs to train the model, alternative to --max_iteration") 
    parser.add_argument("--batch_size", type = int, default = 64, help="Number of example per batch")
    parser.add_argument("--wd", type = float, default = 1e-2, help="weight decay on each layer, default = 0.0 means no decay")

    parser.add_argument("--beta", type = int, default = 500, help="beta to weight between the two losses as in the paper")
    
    parser.add_argument("--train_test_phase", default = 'train', help = 'train, validation or test phase')    
    parser.add_argument("--dataset", default = 'OldHospital', type=str, help ='dataset name to train the network')
    parser.add_argument("--eval_interval_epoch", type = int, default = 100, help="nb of epochs after which the evaluation on validation should take place")

    # specific to this architecture
    parser.add_argument("--nb_run", default = 1, type = int, help ='each run per configuration, useful to compute std')
    parser.add_argument("--experiment", default= 'cambridge', help='group of experiments')
    parser.add_argument("--use_dropout", type = int, default = 0, help='use dropout or not 0 or 1')
    parser.add_argument("--continue_training", type = int, default = 0, help='continue training on the logdir if done.txt is not yet created. Use for resuming training')
    
    FLAGS, unparsed = parser.parse_known_args()

    
    tf.app.run(main= main, argv=[sys.argv[0]] + unparsed)
