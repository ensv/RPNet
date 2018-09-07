'''
Created on Jul 18, 2018

@author: en
'''

import tensorflow as tf
import os
import numpy as np
from sklearn.externals import joblib
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors import OutOfRangeError, InvalidArgumentError

class AbstractNetwork(object):
    '''
    classdocs
    '''

    def __init__(self, setting, flags):


        # optimization params
        self.optimization = flags.optimization
        self.lr = flags.lr
        self.lr_decay_rate = flags.lr_decay_rate
        self.lr_decay_step = flags.lr_decay_step
        self.batch_size = flags.batch_size
        self.train_test_phase= flags.train_test_phase       
        self.nb_epoches = flags.nb_epoches
        self.use_dropout = flags.use_dropout
        
        if hasattr(flags, 'regressor_dimension'):
            self.regressor_dimension = flags.regressor_dimension
        else:
            self.regressor_dimension = 1024
        
        # name of the model, checkpoint, result file
        self.network_name = flags.name
        self.experiment = flags.experiment
        
        self.dataset = flags.dataset
        self.stddev = 5e-2

        self.eval_interval_epoch = flags.eval_interval_epoch
        self.nb_run = flags.nb_run
        
        self.valid_result_fn = 'result_valid'
        self.test_result_fn = 'result_test'
        self.continue_training = flags.continue_training
        
        self.setting = setting
        self.max_iteration = 100000000000000
        self.prefetch_data = 500
        self.eval_interval_epoch = 50
        
        self.asyn_train = False

        self.cambridge_subset = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']

    
    def _try_create_directory(self, path):
        try: os.mkdir(path)
        except: pass
        
    def _train_dir_path(self, suffix = '', lr_factor = 1., data_path_ = None):

        if data_path_ is None:
            path = self.setting.data_path
        else:
            path = data_path_

        path += '/%s/'%self.dataset
        self._try_create_directory(path)

        path += '%s/'%(self.network_name)
        self._try_create_directory(path)

        path += '%s/'%(self.experiment)
        self._try_create_directory(path)

        path += '%s_%d_%0.6f_%0.2f_%d_%d%s'%(self.optimization, self.batch_size, lr_factor*self.lr,
                                                                self.lr_decay_rate, self.lr_decay_step, self.nb_epoches, suffix)

        self._try_create_directory(path)

        for i in range(0, self.nb_run):
            tmp_path = path + '/%d'%i
            if self.train_test_phase == 'train':
                if self.asyn_train:
                    return tmp_path
                if os.path.exists(tmp_path):                    
                    if self.continue_training:
                        if os.path.exists(tmp_path + '/done.txt'):
                            continue
                        else:
                            with open(tmp_path + '/done.txt', 'w') as f:
                                f.write('\n')                                  
                            return tmp_path
                    else:
                        continue
                else:
                    self._try_create_directory(tmp_path)
                    return tmp_path
            else :
                if os.path.exists(tmp_path + '/done.txt') and self.checkpoint_exists(tmp_path):
                    tmp_fn = self.test_result_fn if self.train_test_phase =='test' else self.valid_result_fn

                    print('**********************************')
                    if os.path.exists(tmp_path + '/%s.pkl'%tmp_fn):
                        print('--- \t result file exists')
                        continue
                    else:
                        joblib.dump('1', tmp_path + '/%s.pkl'%tmp_fn, compress=3)
                        return tmp_path
        return ''

    def checkpoint_exists(self, folder):
        for e in os.listdir(folder):
            if 'model.ckpt' in e:
                return True
        return False
        
    def prepare_inference(self):
        raise NotImplementedError
    
    def prepare_loss(self):
        raise NotImplementedError 

    def before_valid_test(self, sess):
        raise NotImplementedError

    def before_train(self, sess):
        raise NotImplementedError

    def _get_train_test_set(self, folder_path = '../data/'):
        raise NotImplementedError 

    def get_train_op(self, nb_replicas = 0):
        
        lr = tf.train.exponential_decay(self.lr, self.global_step, self.lr_decay_step * self.nb_iterations, self.lr_decay_rate, staircase=True)
        tf.summary.scalar('learning_rate', lr)  # @UndefinedVariable
        
        self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('Train/total_loss', self.total_loss)  # @UndefinedVariable
        
        if self.optimization == 'sgd':
            self.optimizer = tf.train.AdamOptimizer()
        elif self.optimization == 'sgdm':
            self.optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov = True)
        elif self.optimization == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(lr)
        else:
            raise NotImplementedError()
        
        if nb_replicas >0:
            self.optimizer = tf.train.SyncReplicasOptimizer(self.optimizer, replicas_to_aggregate=nb_replicas, total_num_replicas=nb_replicas)
        
        grads = self.optimizer.compute_gradients(self.total_loss)
    
        # Apply gradients.
        self.train_op = self.optimizer.apply_gradients(grads, global_step= self.global_step)
    
        # Add histograms for gradients, save only one just to see how small/big it is to adapt the lr
#         for grad, var in grads:
#             if grad is not None:
#                 tf.summary.histogram(var.op.name + '/gradients', grad)  # @UndefinedVariable
#                 break

        self.summary_op = tf.summary.merge_all()  # @UndefinedVariable


    def _get_nb_lines(self):
        
        if type(self.train_filenames) == type([]):
            return len(open(self.train_filenames[0].replace('train.tfrecord', 'train_set.txt'), 'r').readlines())
        
        return len(open(self.train_filenames.replace('train.tfrecord', 'train_set.txt'), 'r').readlines())
    
    
    def prepare_data(self, train_fn, test_valid_fn):

        if self.train_test_phase == 'train':
            dataset = tf.data.TFRecordDataset(self.train_filenames, num_parallel_reads=10)
            dataset = dataset.shuffle(self.setting.dataset_shuffle_size)
            dataset = dataset.map(train_fn, num_parallel_calls = 5)
            dataset = dataset.repeat(self.eval_interval_epoch)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.prefetch_data)
        else:
            dataset = tf.data.TFRecordDataset(self.test_filenames)        
            dataset = dataset.map(test_valid_fn)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.prefetch_data)
            
        valid_dataset = tf.data.TFRecordDataset(self.valid_filenames)        
        valid_dataset = valid_dataset.map(test_valid_fn)
        valid_dataset = valid_dataset.batch(self.batch_size)

        self.iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        self.training_init_op = self.iterator.make_initializer(dataset)
        self.validation_init_op = self.iterator.make_initializer(valid_dataset)

        if hasattr(self, 'use_extraLoss'):
            if self.use_extraLoss:
                self.input_X, self.input_Y, self.extra_input_Y1, self.extra_input_Y2 = self.iterator.get_next()
            else:
                self.input_X, self.input_Y = self.iterator.get_next()
        else:
            self.input_X, self.input_Y = self.iterator.get_next()
        
        self.global_step = tf.Variable(0, dtype='int64', trainable = False, name ='global_step')
        self.nb_iterations = self._get_nb_lines() // self.batch_size
    

    def _custom_evaluation(self, sess):

        # example of prototype
#         loss = []
#         while True:
#             try:
#                 loss.append(sess.run(self.tran_loss))                
#             except OutOfRangeError:
#                 break
#         
#         # validation loss
#         loss = np.array(loss)        
#         joblib.dump((loss, loss.mean()), '%s/%s.pkl'%(self.train_dir, self.test_result_fn), compress=3)
        raise NotImplementedError 
        
    
    def evaluate_test(self):
        
        init = tf.global_variables_initializer()  # @UndefinedVariable
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, 
                                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95)))
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)  # @UndefinedVariable
        
        saver.restore(sess, tf.train.latest_checkpoint(self.train_dir))
        sess.run(self.training_init_op)        
        
        self.before_valid_test(sess)
        self._custom_evaluation(sess)
        sess.close()
        ops.reset_default_graph()
    
    def evaluate_validation(self, sess):
        raise NotImplementedError 
    

    def _define_additional_summaries(self):
        raise NotImplementedError 
    
    
    def _load_pretrained_weight(self, sess):
        raise NotImplementedError 
    
    
    def train(self):
        
        self._define_additional_summaries()        
        init = tf.global_variables_initializer()  # @UndefinedVariable
    
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, 
                                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95)))
        sess.run(init)
    
        self._load_pretrained_weight(sess)
    
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 30)  # @UndefinedVariable
        
        self.summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)    # @UndefinedVariable        

        start_epoch = 0
        if tf.train.latest_checkpoint(self.train_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(self.train_dir))
            start_epoch = int(tf.train.latest_checkpoint(self.train_dir).split('/')[-1].replace('model.ckpt-', ''))
            print ('resuming training from ', start_epoch)
            start_epoch = int(start_epoch)

        iters = sess.run(self.global_step)
        for epoch in range(start_epoch, self.nb_epoches, self.eval_interval_epoch):

            sess.run(self.training_init_op)
            self.before_train(sess)
            # train for half epoch
            while True:
                try:
                    iters += 1
                    if iters % 1000 == 0:
                        try:
                            summary_str, loss = sess.run([self.summary_op, self.total_loss])
                            self.summary_writer.add_summary(summary_str, iters)
                        except InvalidArgumentError: # nan in summary graident
                            self.done_training('nan in loss')
                            sess.close()
                            ops.reset_default_graph()
                            return
                            
                        if np.isnan(loss):
                            self.done_training('nan in loss')
                            sess.close()
                            ops.reset_default_graph()
                            return
                    else:
                        sess.run(self.train_op)
                except OutOfRangeError:
                    break

            if epoch % self.eval_interval_epoch != 0: continue
            self.evaluate_validation(sess)
            saver.save(sess, '%s/model.ckpt'%self.train_dir, global_step=epoch)   
            if iters >= self.max_iteration:

                print('break, iters >= self.max_iteration')
                break
        self.evaluate_validation(sess)        
        saver.save(sess, '%s/model.ckpt'%self.train_dir, global_step=self.nb_epoches)
        
        self.done_training()
        sess.close()
        ops.reset_default_graph()
           
    def done_training(self, message = 'finished'):
        with open('%s/done.txt'%self.train_dir, 'w') as f:
            f.write('done\n')
            f.write(message)
                    
    def process(self):
        
        if self.train_dir == '':
            ops.reset_default_graph()
            print('train_dir = "", default graph has been reset ...')
            return
        print('\n')
        print('train_dir: %s'%self.train_dir)
        
        if self.train_test_phase == 'train':
            self.train()
        elif self.train_test_phase == 'test':
            self.evaluate_test()
        ops.reset_default_graph()
        
    #################################################### Specific to this project ########################################
    def _l2_norm_loss(self, predicts, labels, weights, scope =''):
        if scope == '':
            loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels - predicts), axis=1)))
            tf.add_to_collection('losses', loss*weights)
        else:
            loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels - predicts), axis=1)), name = scope)
            tf.summary.scalar(scope, loss)  # @UndefinedVariable
            tf.add_to_collection('losses', loss*weights)
        return loss * weights
    
    def quat_2_matrix_tf(self, q):
        a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:]
        asq, bsq, csq, dsq = a**2, b**2, c**2, d**2
        ab, ac, ad = 2*a*b, 2*a*c, 2*a*d
        bc, bd = 2*b*c, 2*b*d
        cd = 2*c*d
        mat  = [asq + bsq - csq -dsq,  bc - ad,             bd + ac, \
                bc + ad,               asq -bsq + csq -dsq, cd - ab, \
               bd - ac,                cd + ab,             asq - bsq - csq + dsq]
        
        mat = tf.reshape(tf.concat(mat, axis=1), (-1, 3, 3))
        return mat

    def get_relative_T_in_cam2_ref(self, R2, T1, T2):
        # T1, T2, shape: (batch, 3)
        
        if self.dataset in self.cambridge_subset:
            matrix = self.quat_2_matrix_tf(R2)
            new_c2 = - tf.matmul(matrix, tf.reshape(T2, (-1, 3, 1)))        
            return tf.reshape(tf.matmul(matrix, tf.reshape(T1, (-1, 3, 1))) + new_c2 , (-1, 3, ))  
        elif 'greyc' in self.dataset.lower():
            matrix = self.quat_2_matrix_tf(R2)
            return tf.reshape(tf.matmul(matrix, tf.reshape((T1-T2), (-1, 3, 1)), True), (-1, 3))


    
    def get_quaternion_rotation(self, pose1, pose2, name_):

        if 'greyc' in self.dataset.lower():
            a1, b1, c1, d1 = pose1[:, 0:1], pose1[:, 1:2], pose1[:, 2:3], pose1[:, 3:] 
            a2, b2, c2, d2 = pose2[:, 0:1], -1*pose2[:, 1:2], -1*pose2[:, 2:3], -1*pose2[:, 3:] 
        elif self.dataset in self.cambridge_subset:
            a1, b1, c1, d1 = pose1[:, 0:1], -1*pose1[:, 1:2], -1*pose1[:, 2:3], -1*pose1[:, 3:] 
            a2, b2, c2, d2 = pose2[:, 0:1], pose2[:, 1:2], pose2[:, 2:3], pose2[:, 3:] 
         
        pose = tf.concat( ((a2*a1 - b2*b1 - c2*c1 - d2*d1), 
                           (a2*b1 + b2*a1 + c2*d1 - d2*c1), 
                           (a2*c1 - b2*d1 + c2*a1 + d2*b1), 
                           (a2*d1 + b2*c1 - c2*b1 + d2*a1) ), axis=1, name = name_)
        return pose
    
    def _define_regressor(self, net):
        
        # cls 1
        with tf.variable_scope("cls1"):
            x = tf.layers.flatten(net.layers['cls1_reduction_pose'])
            fc1 = tf.layers.dense(x, self.regressor_dimension, activation=tf.nn.relu, use_bias=True,  # @UndefinedVariable
                    kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                    bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                     kernel_regularizer=l2_regularizer(self.wd),
#                     bias_regularizer=l2_regularizer(self.wd), 
                    name='cls1_fc1')

            
            aux1_xyz_logits = tf.layers.dense(fc1, 3, activation=None, use_bias=True,  # @UndefinedVariable
                                kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                                 kernel_regularizer=l2_regularizer(self.wd),
#                                 bias_regularizer=l2_regularizer(self.wd), 
                                name='cls1_fc1_xyz')
    
            aux1_wpqr_logits = tf.layers.dense(fc1, 4, activation=None, use_bias=True,  # @UndefinedVariable
                                kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                                 kernel_regularizer=l2_regularizer(self.wd),
#                                 bias_regularizer=l2_regularizer(self.wd), 
                                name='cls1_fc1_wpqr')
        # cls 2
        with tf.variable_scope("cls2"):
            x = tf.layers.flatten(net.layers['cls2_reduction_pose'])
            fc1 = tf.layers.dense(x, self.regressor_dimension, activation=tf.nn.relu, use_bias=True,  # @UndefinedVariable
                    kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                    bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                     kernel_regularizer=l2_regularizer(self.wd),
#                     bias_regularizer=l2_regularizer(self.wd), 
                    name='cls2_fc1')

            
            aux2_xyz_logits = tf.layers.dense(fc1, 3, activation=None, use_bias=True,  # @UndefinedVariable
                                kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                                 kernel_regularizer=l2_regularizer(self.wd),
#                                 bias_regularizer=l2_regularizer(self.wd), 
                                name='cls2_fc1_xyz')
    
            aux2_wpqr_logits = tf.layers.dense(fc1, 4, activation=None, use_bias=True,  # @UndefinedVariable
                                kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                                 kernel_regularizer=l2_regularizer(self.wd),
#                                 bias_regularizer=l2_regularizer(self.wd), 
                                name='cls2_fc1_wpqr')        
        # cls 3
        with tf.variable_scope("cls3"):
            x = tf.layers.flatten(net.layers['cls3_pool'])
            fc1 = tf.layers.dense(x, 2*self.regressor_dimension, activation=tf.nn.relu, use_bias=True,  # @UndefinedVariable
                    kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                    bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                     kernel_regularizer=l2_regularizer(self.wd),
#                     bias_regularizer=l2_regularizer(self.wd), 
                    name='cls3_fc1')
            
            xyz_logits = tf.layers.dense(fc1, 3, activation=None, use_bias=True,  # @UndefinedVariable
                                kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                                 kernel_regularizer=l2_regularizer(self.wd),
#                                 bias_regularizer=l2_regularizer(self.wd), 
                                name='cls3_fc1_xyz')
    
            wpqr_logits = tf.layers.dense(fc1, 4, activation=None, use_bias=True,  # @UndefinedVariable
                                kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.01, dtype=tf.float32),
#                                 kernel_regularizer=l2_regularizer(self.wd),
#                                 bias_regularizer=l2_regularizer(self.wd), 
                                name='cls3_fc1_wpqr')
            
        return xyz_logits, wpqr_logits, aux2_xyz_logits, aux2_wpqr_logits, aux1_xyz_logits, aux1_wpqr_logits
