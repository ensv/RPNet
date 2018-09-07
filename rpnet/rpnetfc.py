'''
Created on Jul 20, 2018

@author: en
'''

import argparse, sys, os
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from rpnet import data_preprocessing
from rpnet import RPNet  # @UnresolvedImport
from abstract_network.setting import Setting
from abstract_network.abstract import AbstractNetwork
from GoogLeNet.googlenet_fc_big import GoogLeNetFCB
import tensorflow as tf

slim = tf.contrib.slim  # @UndefinedVariable


class RPNetFC(RPNet, AbstractNetwork):
    '''
    classdocs
    '''


    def __init__(self, flags):
        '''
        Constructor
        '''

        flags.name = 'RPNetFC'
        self.output_dim = 7

        self.wd = flags.wd
        self.beta = flags.beta
        setting = Setting('relative_pose_network')
        AbstractNetwork.__init__(self, setting, flags)
        self.train_dir = self._train_dir_path('_%d_%.4f'%(self.beta, self.wd))
        if self.train_dir == '': return
        self.use_extraLoss = 0
        
        self.eval_interval_epoch = 50
        self.prefetch_data = 500
       
        self._get_train_test_set()
        train_fn, test_fn = data_preprocessing.get_read_tfrecord_fn(self.dataset)
        self.prepare_data(train_fn, test_fn)
        self.prepare_inference()
        self.prepare_loss()
        self.get_train_op()
        
        self.max_iteration = 750000

    # define the metric network architecture
    def _metric_nework(self, net1, net2, scope):
        
        x = tf.concat((net1, net2), axis = 1 , name = '%s_fusion'%scope)

        fc2 = tf.layers.dense(x, 128, activation=tf.nn.relu, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='%s_fc2'%scope)
        
        
        tran = tf.layers.dense(fc2, 3, activation=None, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='%s_fc3_translation'%scope)

        rotation = tf.layers.dense(fc2, 4, activation=None, use_bias=True,  # @UndefinedVariable
                            kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                            bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
                            kernel_regularizer=l2_regularizer(self.wd),
                            bias_regularizer=l2_regularizer(self.wd), 
                            name='%s_fc3_rotation'%scope)
        

        return tran, rotation


    def prepare_inference(self):
        
        with tf.variable_scope("siamese") as scope:
            net_1 = GoogLeNetFCB({'data': self.input_X[:, 0]})
            scope.reuse_variables()
            net_2 = GoogLeNetFCB({'data': self.input_X[:, 1]})
        
        weight_vars = [v for v in tf.global_variables() if "weights" in v.name]
        print('weight var: ', len(weight_vars))
        
        if self.wd > 0:
            for v in tf.global_variables():
                if "weights" in v.name:
                    tmp_loss = tf.nn.l2_loss(v) * self.wd
                    tf.add_to_collection('losses', tmp_loss)        
        

        self.xyz_logits, self.wpqr_logits = self._metric_nework(net_2.layers['cls3_fc1'], net_1.layers['cls3_fc1'], scope = 'cls3')
        self.aux1_xyz_logits, self.aux1_wpqr_logits = self._metric_nework(net_2.layers['cls1_fc1'], net_1.layers['cls1_fc1'], scope = 'cls1')
        self.aux2_xyz_logits, self.aux2_wpqr_logits = self._metric_nework(net_2.layers['cls2_fc1'], net_1.layers['cls2_fc1'], scope = 'cls2')

        self.net_1 = net_1
        self.net_2 = net_2

         
def main(_):
    
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if FLAGS.experiment == 'cambridge':

        for dataset in ['KingsCollege', 'OldHospital', 'StMarysChurch', 'ShopFacade']:
            for lr in [1e-3]:
                for wd in [1e-2]:
                    for tmp_beta in [10]:
                        FLAGS.wd = wd
                        FLAGS.lr = lr
                        FLAGS.beta = tmp_beta
                        FLAGS.dataset = dataset
                        net = RPNetFC(FLAGS)
                        net.process()
                        
    else:
        print ('unknown experimentation')
        raise NotImplementedError
    os.chdir(working_dir)


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.register("type", "bool", lambda v: v.lower() == "true")
    
    parser.add_argument("--optimization", default= 'sgdm', type=str, help= "use adadelta, sgd, sgdm")
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Initial learning rate")
    parser.add_argument("--lr_decay_rate", type = float, default = 0.90, help = 'decay rate for learning rate')
    parser.add_argument("--lr_decay_step", type = int, default = 80, help = 'step to decay learning rate, use big number of step to avoid decaying the lr')
    parser.add_argument("--nb_epoches", type = int, default = 1000, help = "Nb of epoches to train the model, alternative to --max-step") 
    parser.add_argument("--batch_size", type = int, default = 32, help="Number of example per batch")
    parser.add_argument("--wd", type = float, default = 1e-2, help="weight decay on each layer, default = 0.0 mean no decay")

    parser.add_argument("--beta", type = int, default = 1, help="beta to weight between the two losses as in the paper")
    
    parser.add_argument("--train_test_phase", default = 'train', help = 'train, validation or test')    
    parser.add_argument("--dataset", default = 'OldHospital', type=str, help ='dataset name to train the network')
    parser.add_argument("--eval_interval_epoch", type = int, default = 100, help="nb of epoches after which the evaluation on validation set takes place")

    # specific to this architecture
    parser.add_argument("--nb_run", default = 1, type=int, help ='each run per configuration')
    parser.add_argument("--experiment", default= 'cambridge', help='group of experiments')
    parser.add_argument("--use_dropout", type = int, default = 0, help='use dropout or not 0 or 1')
    parser.add_argument("--continue_training", type = int, default = 0, help='continue training on the logdir if done.txt is not yet created')

    
    FLAGS, unparsed = parser.parse_known_args()

    
    tf.app.run(main= main, argv=[sys.argv[0]] + unparsed)
