'''
Created on Jul 20, 2018

@author: en
'''

import argparse, sys, os
from tensorflow.python.framework.errors_impl import OutOfRangeError
import math
from sklearn.externals import joblib
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import data_preprocessing  # @UnresolvedImport
from abstract_network.setting import Setting
from abstract_network.abstract import AbstractNetwork
from GoogLeNet.googlenet import GoogLeNet
import tensorflow as tf
import numpy as np


slim = tf.contrib.slim  # @UndefinedVariable



class RPNet(AbstractNetwork):
    '''
    classdocs
    '''


    def __init__(self, flags):
        '''
        Constructor
        '''
        
        flags.name = 'RPNet'
        self.output_dim = 7

        self.use_extraLoss = flags.use_extraLoss
        self.wd = flags.wd
        self.beta = flags.beta
        setting = Setting('relative_pose_network')
        AbstractNetwork.__init__(self, setting, flags)
        self.train_dir = self._train_dir_path('_%d_%.4f_%d'%(self.beta, self.wd, self.use_extraLoss))
        if self.train_dir == '': return
        
        self.eval_interval_epoch = 50
        self.prefetch_data = 500

        self._get_train_test_set()
        train_fn, test_fn = data_preprocessing.get_read_tfrecord_fn(self.dataset, self.use_extraLoss)
        self.prepare_data(train_fn, test_fn)
        self.prepare_inference()
        self.prepare_loss()
        self.get_train_op()
        
        self.max_iteration = 500000

    
    def _get_train_test_set(self):
        
        if self.dataset in self.cambridge_subset:
            path = self.setting.data_path.replace('relative_pose_network', 'relative_cambridge') + '/' + self.dataset
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
        
        with tf.variable_scope("siamese") as scope:
            net_1 = GoogLeNet({'data': self.input_X[:, 0]})
            scope.reuse_variables()
            net_2 = GoogLeNet({'data': self.input_X[:, 1]})
        
        weight_vars = [v for v in tf.global_variables() if "weights" in v.name]
        print('weight var: ', len(weight_vars))
        
        if self.wd > 0:
            for v in tf.global_variables():
                if "weights" in v.name:
                    tmp_loss = tf.nn.l2_loss(v) * self.wd
                    tf.add_to_collection('losses', tmp_loss)        
        
        self.xyz_logits = self.get_relative_T_in_cam2_ref(net_2.layers['cls3_fc_pose_wpqr'], net_1.layers['cls3_fc_pose_xyz'], net_2.layers['cls3_fc_pose_xyz'])
        self.wpqr_logits = self.get_quaternion_rotation(net_1.layers['cls3_fc_pose_wpqr'],  net_2.layers['cls3_fc_pose_wpqr'], name_='cls3_pose_wpqr')
        self.aux1_xyz_logits = self.get_relative_T_in_cam2_ref(net_2.layers['cls1_fc_pose_wpqr'], net_1.layers['cls1_fc_pose_xyz'], net_2.layers['cls1_fc_pose_xyz'])
        self.aux1_wpqr_logits = self.get_quaternion_rotation(net_1.layers['cls1_fc_pose_wpqr'], net_2.layers['cls1_fc_pose_wpqr'], name_='cls1_pose_wpqr')
        self.aux2_xyz_logits = self.get_relative_T_in_cam2_ref(net_2.layers['cls2_fc_pose_wpqr'], net_1.layers['cls2_fc_pose_xyz'], net_2.layers['cls2_fc_pose_xyz'])
        self.aux2_wpqr_logits = self.get_quaternion_rotation(net_1.layers['cls2_fc_pose_wpqr'], net_2.layers['cls2_fc_pose_wpqr'], name_='cls2_pose_wpqr')
        
        self.net_1 = net_1
        self.net_2 = net_2
         
    def extra_loss_4(self, net, pose):
        self._l2_norm_loss(net.layers['cls3_fc_pose_xyz'], labels=pose[:, :3], weights=self.translation_weight, scope='Train/Extra_1_translation_loss')
        self._l2_norm_loss(net.layers['cls3_fc_pose_wpqr'], labels=pose[:, 3:], weights=self.beta, scope='Train/Extra_1_rotation_loss')

        # Aux classifiers
        self._l2_norm_loss(net.layers['cls2_fc_pose_xyz'], labels=pose[:, :3], weights=self.translation_weight/ 3.33)
        self._l2_norm_loss(net.layers['cls2_fc_pose_wpqr'], labels=pose[:, 3:], weights=self.beta/3.33)
        self._l2_norm_loss(net.layers['cls1_fc_pose_xyz'], labels=pose[:, :3], weights=self.translation_weight/3.33)
        self._l2_norm_loss(net.layers['cls1_fc_pose_wpqr'], labels=pose[:, 3:], weights=self.beta/3.33)

    def prepare_loss(self):
        
        self.translation_weight = 1
        
        self.tran_loss = self._l2_norm_loss(self.xyz_logits, labels=self.input_Y[:, :3], weights=self.translation_weight, scope='Train/translation_loss')
        self.rotat_loss = self._l2_norm_loss(self.wpqr_logits, labels=self.input_Y[:, 3:], weights=self.beta, scope='Train/rotation_loss')

        # Aux classifiers
        self.tran_loss2 = self._l2_norm_loss(self.aux2_xyz_logits, labels=self.input_Y[:, :3], weights=self.translation_weight/3.33, scope='Train/aux2_translation_loss')
        self.rotat_loss2 = self._l2_norm_loss(self.aux2_wpqr_logits, labels=self.input_Y[:, 3:], weights=self.beta/3.33, scope='Train/aux2_rotation_loss')
        self.tran_loss1 = self._l2_norm_loss(self.aux1_xyz_logits, labels=self.input_Y[:, :3], weights=self.translation_weight/3.33, scope='Train/aux1_translation_loss')
        self.rotat_loss1 = self._l2_norm_loss(self.aux1_wpqr_logits, labels=self.input_Y[:, 3:], weights=self.beta/3.33, scope='Train/aux1_rotation_loss')

        if self.use_extraLoss:
            self.extra_loss_4(self.net_1, self.extra_input_Y1)
            self.extra_loss_4(self.net_2, self.extra_input_Y2)

    def _load_pretrained_weight(self, sess):
        self.net_1.load('../GoogLeNet/weights/posenet.npy', sess, 'siamese/')

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
                tmp[3] = tmp[3]/ np.linalg.norm(tmp[3], axis=1, keepdims = True)
                d = np.abs(np.sum(np.multiply(tmp[3], tmp[4][:, 3:]), axis=1))
                np.putmask(d, d> 1, 1)
                np.putmask(d, d< -1, -1)
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
                
                tmp[3] = tmp[3]/ np.linalg.norm(tmp[3], axis=1, keepdims=True)
                d = np.abs(np.sum(np.multiply(tmp[3], tmp[4][:, 3:]), axis=1))
                np.putmask(d, d> 1, 1)
                np.putmask(d, d< -1, -1)
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
        loss = np.array(loss)
        np.save('%s/loss'%self.train_dir, pose)
        
        # validation error
        error = np.array([np.concatenate(error[0]), np.concatenate(error[1]), np.concatenate(error[2])])        
        joblib.dump(error, '%s/%s.pkl'%(self.train_dir, self.test_result_fn), compress=3)    
    
def main(_):
    
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if FLAGS.experiment == 'cambridge':

        for dataset in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
            for lr in [1e-5]:
                for tmp_beta in [1]:
                    FLAGS.beta = tmp_beta
                    FLAGS.lr = lr
                    FLAGS.dataset = dataset            
                    net = RPNet(FLAGS)
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

    parser.add_argument("--beta", type = int, default = 1, help="beta to weight between the two losses as in the paper")
    
    parser.add_argument("--train_test_phase", default = 'train', help = 'train, validation or test phase')    
    parser.add_argument("--dataset", default = 'OldHospital', type=str, help ='dataset name to train the network')
    parser.add_argument("--eval_interval_epoch", type = int, default = 100, help="nb of epochs after which the evaluation on validation should take place")

    # specific to this architecture
    parser.add_argument("--nb_run", default = 1, type = int, help ='each run per configuration, useful to compute std')
    parser.add_argument("--experiment", default= 'cambridge', help='group of experiments')
    parser.add_argument("--use_dropout", type = int, default = 0, help='use dropout or not 0 or 1')
    parser.add_argument("--continue_training", type = int, default = 0, help='continue training on the logdir if done.txt is not yet created. Use for resuming training')
    parser.add_argument("--use_extraLoss", type = int, default = 0, help='use extraLoss for absolute pose')

    
    FLAGS, unparsed = parser.parse_known_args()

    
    tf.app.run(main= main, argv=[sys.argv[0]] + unparsed)
