# Wind speed prediction, LSTM model, zty

################################### import ###################################
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

################################## constant ##################################
train_x_path_ = './train_x.dat'
train_y_path_ = './train_y.dat'
test_x_path_ = './test_x.dat'
test_y_path_ = './test_y.dat'
model_path_ = './model/'  # the path to save model params

train_size_ = 10000  # training dataset size
test_size_ = 1556   # test dataset size
x_dim_ = 6    # dimension of x in every time step
y_dim_ = 6    # dimension of y
seq_len_ = 60  # LSTM BP truncate length, length of [x_1, ..., x_seq_len]

toPlot_ = 'y'
toRestore_ = 'n'

batch_size_ = 50
num_iters_ = 10000  # training iterations
learning_rate_ = 0.1
h_size_ = 50  # LSTM hidden state vector size

#################################### data ####################################
def construct_simulate_data():
    cnt = 0
    train_x = np.zeros((train_size_, seq_len_, x_dim_))
    for i in range(train_size_):
        for j in range(seq_len_):
            for k in range(x_dim_):
                train_x[i][j][k] = cnt
                cnt += 1
    train_y = np.zeros((train_size_, y_dim_))
    
    for i in range(train_size_):
        for j in range(y_dim_):
            train_y[i][j] = sum(sum(train_x[i])) + j
    
    test_x = np.zeros((test_size_, seq_len_, x_dim_))
    for i in range(test_size_):
        for j in range(seq_len_):
            for k in range(x_dim_):
                test_x[i][j][k] = cnt
                cnt += 1
    test_y = np.zeros((test_size_, y_dim_))
    
    for i in range(test_size_):
        for j in range(y_dim_):
            test_y[i][j] = sum(sum(test_x[i])) + j
    
    train_x.tofile(train_x_path_)
    train_y.tofile(train_y_path_)
    test_x.tofile(test_x_path_)
    test_y.tofile(test_y_path_)    
    print('Simulated data constructed and saved to file.')
        
        
class data:
    
    def __init__(self, x, y):
        self.x = np.zeros((seq_len_, x_dim_))
        self.y = np.zeros((y_dim_))          
        for i in range(seq_len_):
            for j in range(x_dim_):
                self.x[i][j] = x[i][j]
        for i in range(y_dim_):
                self.y[i] = y[i]           
    
    def show(self):
        print('Data show called, x:', self.x, 'y:', self.y)
    

class dataset:
    
    def __init__(self, train_or_test):  # train_or_test = 'train' or 'test' 
        self.set = []
        self.train_or_test = train_or_test
        self.idx = 0    # for choosing batch
        if self.train_or_test == 'train':
            self.size = train_size_
            self.xpath = train_x_path_
            self.ypath = train_y_path_
        else:
            self.size = test_size_
            self.xpath = test_x_path_
            self.ypath = test_y_path_
        read_x = np.fromfile(self.xpath, dtype = np.float64)
        read_y = np.fromfile(self.ypath, dtype = np.float64)
        read_x = read_x.reshape((self.size, seq_len_, x_dim_))
        read_y = read_y.reshape((self.size, y_dim_))
        '''
        # for debugging
        print(read_x)
        print(read_y)
        '''
        for i in range(self.size):
            tmp = data(read_x[i], read_y[i])
            self.set.append(tmp)
        print('Dataset constructed.')
    
    def show(self):
        for tmp in self.set:
            tmp.show()
        print('Dataset shown.')
    
    def shuffle(self):
        '''
        perm = np.arange(self.size)
        np.random.shuffle(perm)
        tmp = []
        for i in range(self.size):
            tmp.append(self.set[perm[i]])
        self.set = tmp
        '''
        return 0
    
    def get_batch(self):
        start = self.idx
        self.idx += batch_size_
        if self.idx > self.size:
            self.shuffle()
            start = 0
            self.idx = batch_size_
        end = self.idx
        # seperate into different slices 
        batch_x = []
        batch_y = []
        for i in range(start, end):
            batch_x.append(self.set[i].x)
            batch_y.append(self.set[i].y)
        return(batch_x, batch_y)
    
        
################################### model ####################################
# for training process visualisation
def plot(loss_list):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    plt.draw()
    plt.pause(0.0001)
    
# regularization
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev = stddev))
    if(wl is not None):
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# regularized loss
def regularized_loss(loss):
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

# LSTM
def Graph(trainset, testset):
    # defining weight matrixes and vectors
    X = tf.placeholder(tf.float32, [batch_size_, seq_len_, x_dim_])
    Y = tf.placeholder(tf.float32, [batch_size_, y_dim_])
    
    init_c = tf.placeholder(tf.float32, [batch_size_, h_size_])
    current_c = tf.placeholder(tf.float32, [batch_size_, h_size_])
    next_c = tf.placeholder(tf.float32, [batch_size_, h_size_])
    init_h = tf.placeholder(tf.float32, [batch_size_, h_size_])
    current_h = tf.placeholder(tf.float32, [batch_size_, h_size_])
    next_h = tf.placeholder(tf.float32, [batch_size_, h_size_])

    # LSTM
    Wf = variable_with_weight_loss(shape = [h_size_ + x_dim_, h_size_], stddev = 3e-2, wl = 0.000025)
    bf = variable_with_weight_loss(shape = [1, h_size_], stddev = 3e-2, wl = 0.000025)
    Wi = variable_with_weight_loss(shape = [h_size_ + x_dim_, h_size_], stddev = 3e-2, wl = 0.000025)
    bi = variable_with_weight_loss(shape = [1, h_size_], stddev = 3e-2, wl = 0.000025)
    Wc = variable_with_weight_loss(shape = [h_size_ + x_dim_, h_size_], stddev = 3e-2, wl = 0.000025)
    bc = variable_with_weight_loss(shape = [1, h_size_], stddev = 3e-2, wl = 0.000025)
    Wo = variable_with_weight_loss(shape = [h_size_ + x_dim_, h_size_], stddev = 3e-2, wl = 0.000025)
    bo = variable_with_weight_loss(shape = [1, h_size_], stddev = 3e-2, wl = 0.000025)   
    # output
    Wout = variable_with_weight_loss(shape = [h_size_, y_dim_], stddev = 3e-2, wl = 0.000025)
    bout = variable_with_weight_loss(shape = [1, y_dim_], stddev = 3e-2, wl = 0.000025)

    # Unpack columns
    input_series = tf.unstack(X, axis = 1)    

    # Forward pass for training
    current_h = init_h
    current_c = init_c
    for i in range(seq_len_):
        # input
        current_input = tf.reshape(input_series[i], [batch_size_, x_dim_])
       
        # LSTM
        h_concat_x = tf.concat([current_input, current_h], 1)
        f = tf.nn.sigmoid(tf.matmul(h_concat_x, Wf) + bf)
        i = tf.nn.sigmoid(tf.matmul(h_concat_x, Wi) + bi)
        c = tf.tanh(tf.matmul(h_concat_x, Wc) + bc)
        next_c = tf.multiply(current_c, f) + tf.multiply(c, i)
        o = tf.nn.sigmoid(tf.matmul(h_concat_x, Wo) + bo)
        next_h = tf.multiply(tf.tanh(next_c), o)
        
        # renew
        current_h = next_h
        current_c = next_c
    
    # output given output h
    ypred = tf.matmul(current_h, Wout) + bout

    # for training process
    losses = tf.abs(ypred - Y)
    total_loss = tf.reduce_mean(losses)
    total_loss = regularized_loss(total_loss)
    
    # optimizer
    train_step = tf.train.AdagradOptimizer(learning_rate_)

    # gradient clipping
    grads_and_vars = train_step.compute_gradients(total_loss, [Wf, bf, Wi, bi, Wc, bc, Wo, bo, Wout, bout])
    capped_grads_and_vars = [(tf.clip_by_norm(gv[0], 5), gv[1]) for gv in grads_and_vars]
    train_step.apply_gradients(capped_grads_and_vars)
    train_step = train_step.minimize(total_loss)
    
    # for model saving
    saver = tf.train.Saver() 

    # training session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
            
        if toPlot_ == 'y':
            plt.ion()
            plt.figure()
            plt.show()

        loss_list = []
        
        if toRestore_ == 'y':
            saver.restore(sess, model_path_)

        for epoch_idx in range(num_iters_):
            _current_h = np.zeros((batch_size_, h_size_))
            _current_c = np.zeros((batch_size_, h_size_)) 
            
            train_step.run(  # sess.run not BPing @ CVDA???
                    feed_dict = {
                    X: trainset.get_batch()[0],
                    Y: trainset.get_batch()[1],
                    init_h: _current_h,
                    init_c: _current_c
                    })
    
            _total_loss, _ypred = sess.run(
                    [total_loss, ypred],
                    feed_dict = {
                    X: testset.get_batch()[0],
                    Y: testset.get_batch()[1],
                    init_h: _current_h,
                    init_c: _current_c
                    })

            loss_list.append(_total_loss)

            if epoch_idx % 50 == 0:
               print("Step", epoch_idx, "Loss", _total_loss)
               if(toPlot_ == 'y'):
                   plot(loss_list)    
               # save
               saver.save(sess, model_path_)
               print('saved')
               


#################################### main ####################################
#construct_simulate_data()
trainset = dataset('train')
testset = dataset('test')
Graph(trainset, testset)
