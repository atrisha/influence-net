'''
Created on Feb 1, 2017

@author: atri
'''

import tensorflow as tf
import numpy as np
import time
import sys
import flags
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import env_summary

GRID_SIZE_ROWS = flags.GRID_SIZE_ROWS
GRID_SIZE_COLS = flags.GRID_SIZE_COLS
NUM_INPUT_CHANNELS = flags.NUM_INPUT_CHANNELS
NUM_ENV_CHANNELS = flags.NUM_ENV_CHANNELS
NUM_OUTPUT_CHANNELS = flags.NUM_OUTPUT_CHANNELS
BATCH_SIZE = flags.BATCH_SIZE
DECAY_STEP = flags.DECAY_STEP
NUM_EPOCHS = flags.NUM_EPOCHS
SEED = flags.SEED
EVAL_BATCH_SIZE = flags.EVAL_BATCH_SIZE
EVAL_FREQUENCY = flags.EVAL_FREQUENCY


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def main(_):
    num_epochs = NUM_EPOCHS
    '''here : train_data_env_node = tf.placeholder(tf.float32, [GRID_SIZE_ROWS,GRID_SIZE_COLS,NUM_INPUT_CHANNELS])'''
    data_env_node = tf.placeholder(tf.float32, [1,NUM_ENV_CHANNELS])
    data_ego_node = tf.placeholder(tf.float32, [NUM_INPUT_CHANNELS,1])
    target_node = tf.placeholder(tf.float32, [NUM_OUTPUT_CHANNELS,1])
    
    def model(data_env,data_ego):
    #def model(data_ego):    
        #data_env = tf.nn.l2_normalize(data_env, dim=[0,1])
        
        input_layer = tf.reshape(data_env, [NUM_ENV_CHANNELS, 1])
        
        with tf.variable_scope('hidden1') as scope:
            #weight_layer_1 = tf.Variable(tf.random_gamma(shape = [NUM_ENV_CHANNELS,10], alpha = 1.0, beta = 0.5))
            weight_layer_1 = tf.Variable(tf.random_normal(shape = [NUM_ENV_CHANNELS,10]))
            variable_summaries(weight_layer_1)
            biases_layer_1 = tf.Variable(tf.fill([10], 0.1))
            hidden_layer = tf.nn.tanh(tf.matmul(tf.transpose(input_layer), weight_layer_1) + biases_layer_1, name=scope.name)
        
        with tf.variable_scope('output1') as scope:
            weight_layer_2 = tf.Variable(tf.random_normal([10,3]))
            biases_layer_1 = tf.Variable(tf.zeros([1,3], dtype=tf.float32))
            output_1 = tf.transpose(tf.nn.tanh(tf.matmul(hidden_layer, weight_layer_2) + biases_layer_1, name=scope.name)) # 5 X 1
            tf.summary.histogram('activations', output_1)
            
        ''' data_ego should be in shape 7 X 1'''
            
        input_2 = tf.concat(0,[data_ego,output_1]) # 9 X 1
        
        
        with tf.variable_scope('hidden2') as scope:
            weight_layer_3 = tf.Variable(tf.random_normal([10, 20]))
            biases_layer_3 = tf.Variable(tf.zeros([20], dtype=tf.float32))
            hidden_layer_3 = tf.nn.tanh(tf.matmul(tf.transpose(input_2), weight_layer_3) + biases_layer_3, name=scope.name)
            
        with tf.variable_scope('hidden3') as scope:
            weight_layer_5 = tf.Variable(tf.random_normal([20, 10]))
            biases_layer_5 = tf.Variable(tf.zeros([10], dtype=tf.float32))
            hidden_layer_5 = tf.nn.tanh(tf.matmul(hidden_layer_3, weight_layer_5) + biases_layer_5, name=scope.name)
            
        with tf.variable_scope('output2') as scope:    
            weight_layer_4 = tf.Variable(tf.random_normal([10,2]))
            biases_layer_4 = tf.Variable(tf.zeros([1,2], dtype=tf.float32))
            output_2 = tf.transpose(tf.matmul(hidden_layer_5, weight_layer_4) + biases_layer_4, name=scope.name) # 2 X 1
            
        return output_2
    
    output_vector = model(data_env_node, data_ego_node)
    #output_vector = model(train_data_ego_node)
    diff = tf.subtract(target_node,output_vector)
    l2_loss = tf.reduce_sum(tf.square(diff) , 1)
    
    batch = tf.Variable(0, dtype=tf.float32)
    
    learning_rate = tf.train.exponential_decay(
      .01,                # Base learning rate.
      batch,  # Current index into the dataset.
      DECAY_STEP,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
    
    '''optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(l2_loss,
                                                       global_step=batch)'''
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss,global_step=batch)
    #eval_prediction = model(eval_data_env_node,eval_data_ego_node)
    merged = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    start_time = time.time()
    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter('./train')
        print('Initialized!')
        
        saver.restore(sess, "./model.ckpt")
        print("Model restored.")
        eval_loss = []
        env_eval_data_batch, ego_eval_data_batch,target_eval_data_batch = env_summary.get_env(True)
        for step in range(env_eval_data_batch.shape[0]):
            feed_dict = {data_env_node: env_eval_data_batch[step],
                       data_ego_node: ego_eval_data_batch[step],
                       target_node : target_eval_data_batch[step]}
            if step % 1000 == 0:
                summary,_,loss_val,lr,output = sess.run([merged,optimizer,l2_loss,learning_rate,output_vector], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                train_writer.flush()
            else:
                _,loss_val,lr,output = sess.run([optimizer,l2_loss,learning_rate,output_vector], feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            eval_loss.append(loss_val)
            if step % 1000 == 0:
            #print('Step :', loop,'-',step , 'time_lapse' , 1000 * elapsed_time )
                print('Minibatch loss:',loss_val,' Learning rate:',lr)
                print('input was :',ego_eval_data_batch[step])
                print('predicted :', output,' ; actual:',target_eval_data_batch[step])
                print('--')
            #print('x_a,y_a (predicted ):',output,' | (actual)', target_data_batch[step])
            #print('--')
        res_str = 'loss statistics after run :', 'mean=',np.mean(eval_loss),' median=',np.median(eval_loss),' max=',np.max(eval_loss),' min=',np.min(eval_loss) 
        print(res_str)
        sess.close()
    return res_str
            
if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])        
        
