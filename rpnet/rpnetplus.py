'''
Created on Jul 20, 2018

@author: en
'''

import argparse, sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from rpnet import RPNet, data_preprocessing  # @UnresolvedImport
from abstract_network.setting import Setting
from abstract_network.abstract import AbstractNetwork
import tensorflow as tf



slim = tf.contrib.slim  # @UndefinedVariable

class RPNetPlus(RPNet, AbstractNetwork):
    '''
    classdocs
    '''


    def __init__(self, flags):
        '''
        Constructor
        '''
        
        flags.name = 'RPNetPlus'
        self.output_dim = 7

        self.use_extraLoss = flags.use_extraLoss
        self.wd = flags.wd
        self.beta = flags.beta
        self.alpha = flags.alpha
        self.gamma = flags.gamma
        self.use_aux = flags.use_aux
        setting = Setting('relative_pose_network')
        AbstractNetwork.__init__(self, setting, flags)
        self.train_dir = self._train_dir_path('_%d_%.4f_%d_%d_%d_%d'%(self.beta, self.wd, self.use_extraLoss, self.gamma, self.alpha, self.use_aux))
        if self.train_dir == '': return
        
        self.eval_interval_epoch = 50
        self.prefetch_data = 500
       
        self._get_train_test_set()
        train_fn, test_fn = data_preprocessing.get_read_tfrecord_fn(self.dataset, self.use_extraLoss)
        self.prepare_data(train_fn, test_fn)
        self.prepare_inference()
        self.prepare_loss()
        self.get_train_op()
        
        self.max_iteration = 200000
    
    def extra_loss_4(self, net, pose):
        self._l2_norm_loss(net.layers['cls3_fc_pose_xyz'], labels=pose[:, :3], weights=1., scope='Train/Extra_1_translation_loss')
        self._l2_norm_loss(net.layers['cls3_fc_pose_wpqr'], labels=pose[:, 3:], weights=self.beta, scope='Train/Extra_1_rotation_loss')

        if self.use_aux:
            # Aux classifiers
            self._l2_norm_loss(net.layers['cls2_fc_pose_xyz'], labels=pose[:, :3], weights=1./ 3.33)
            self._l2_norm_loss(net.layers['cls2_fc_pose_wpqr'], labels=pose[:, 3:], weights=self.beta/3.33)
            self._l2_norm_loss(net.layers['cls1_fc_pose_xyz'], labels=pose[:, :3], weights=1./3.33)
            self._l2_norm_loss(net.layers['cls1_fc_pose_wpqr'], labels=pose[:, 3:], weights=self.beta/3.33)

    def prepare_loss(self):
        
        self.translation_weight = self.gamma
        
        self.tran_loss = self._l2_norm_loss(self.xyz_logits, labels=self.input_Y[:, :3], weights=self.translation_weight, scope='Train/translation_loss')
        self.rotat_loss = self._l2_norm_loss(self.wpqr_logits, labels=self.input_Y[:, 3:], weights=self.alpha, scope='Train/rotation_loss')

        if self.use_aux:
            # Aux classifiers
            self.tran_loss2 = self._l2_norm_loss(self.aux2_xyz_logits, labels=self.input_Y[:, :3], weights=self.translation_weight/3.33, scope='Train/aux2_translation_loss')
            self.rotat_loss2 = self._l2_norm_loss(self.aux2_wpqr_logits, labels=self.input_Y[:, 3:], weights=self.alpha/3.33, scope='Train/aux2_rotation_loss')
            self.tran_loss1 = self._l2_norm_loss(self.aux1_xyz_logits, labels=self.input_Y[:, :3], weights=self.translation_weight/3.33, scope='Train/aux1_translation_loss')
            self.rotat_loss1 = self._l2_norm_loss(self.aux1_wpqr_logits, labels=self.input_Y[:, 3:], weights=self.alpha/3.33, scope='Train/aux1_rotation_loss')

        if self.use_extraLoss:
            self.extra_loss_4(self.net_1, self.extra_input_Y1)
            self.extra_loss_4(self.net_2, self.extra_input_Y2)
   
def main(_):
    
    working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if FLAGS.experiment == 'cambridge':
                
        for dataset in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
            for lr in [1e-5]:
                FLAGS.lr = lr
                FLAGS.dataset = dataset
                net = RPNetPlus(FLAGS)
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
    parser.add_argument("--use_extraLoss", type = int, default = 1, help='use extraLoss for absolute pose')
    parser.add_argument("--gamma", type = int, default = 10, help='weight for relative translation loss')
    parser.add_argument("--alpha", type = int, default = 100, help='weight for relative rotation loss')
    parser.add_argument("--use_aux", type = int, default = 0, help = "use auxilary loss of googLeNet or not")
    
    FLAGS, unparsed = parser.parse_known_args()

    
    tf.app.run(main= main, argv=[sys.argv[0]] + unparsed)
