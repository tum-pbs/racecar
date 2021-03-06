import tensorflow as tf
import numpy as np
import os
import time
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--training", type=int, default=1, help="training or testing")
parser.add_argument("--testdir", type=str, default=None, help="specify log file dir")
parser.add_argument("--testnum", type=int, default=-1, help="specify file name")
parser.add_argument("--modelnum", type=int, default=-1, help="specify model name")
parser.add_argument("--basePath", type=str, default="", help="specify base path")
parser.add_argument("--batchsize", type=int, default=64, help="set batch size")
parser.add_argument("--epochnum", type=int, default=100, help="set training epochs")
parser.add_argument("--learningrate", type=float, default=0.0001, help="set learning rate")
parser.add_argument("--maxsave", type=int, default=5, help="set saving number")
parser.add_argument("--rrfactor", type=float, default=0.0, help="set factor for rr term")
parser.add_argument("--orthofactor", type=float, default=0.0, help="set factor for orthogonal term")
parser.add_argument("--runfile", type=str, default="run.py", help="specify run file for copy")
args = parser.parse_args()
if (not args.training):
    if args.testnum < 0 or args.modelnum < 0:
        print("specify --testnum and --modelnum for testing!")
        exit()

if args.testdir:
    folderpre = args.testdir
else:
    folderpre = "default"

BATCH_SIZE = 2
if not args.training:
    BATCH_SIZE = 1
CLASS_NUM = 10
EPOCHS = args.epochnum
learningratevalue = args.learningrate
maxToKeep = args.maxsave
epsilon = 1e-6
imagewidth = 28
imageheight = 28


def makedir():
    count = 0
    currentdir = os.getcwd()+"/"
    while os.path.exists(args.basePath+folderpre+"/test_%04d/"%count):
        count += 1
    targetdir = args.basePath+folderpre+"/test_%04d/"%count
    os.makedirs(targetdir)
    return targetdir


test_path = makedir()
testf = open(test_path + "testaccuracy.txt",'a+')
trainf = open(test_path + "trainloss.txt",'a+')
timef = open(test_path + "elapsedtime.txt",'a+')
os.system("cp %s %s/%s"%(args.runfile,test_path,args.runfile))
os.system("cp %s %s/%s"%(__file__,test_path,__file__))

# training data
num1, num2 = 0,1
x_train0 = np.reshape(imageio.imread("MNIST/%d.png"%num1),[1,imagewidth*imageheight])
x_train1 = np.reshape(imageio.imread("MNIST/%d.png"%num2),[1,imagewidth*imageheight])
y_train0 = np.zeros([1,10])
y_train1 = np.zeros([1,10])
y_train0[0,num1]=1
y_train1[0,num2]=1
x_train = np.concatenate((x_train0,x_train1),axis=0)
y_train = np.concatenate((y_train0,y_train1),axis=0)

# testing data
x_test0 = np.reshape(imageio.imread("MNIST/%d_test.png"%num1),[1,imagewidth*imageheight])
x_test1 = np.reshape(imageio.imread("MNIST/%d_test.png"%num2),[1,imagewidth*imageheight])
x_test = np.concatenate((x_test0,x_test1),axis=0)
y_test = y_train

TOTALWEIGHT = 0
def weight_variable(name, shape):
    var = tf.get_variable(name,shape,initializer = tf.glorot_uniform_initializer())
    global TOTALWEIGHT
    if len(shape) == 4:
        print("Convolutional layer: {}".format(shape))
        TOTALWEIGHT += shape[0]*shape[1]*shape[2]*shape[3]
    if len(shape) == 2:
        print("fully connected layer: {}".format(shape))
        TOTALWEIGHT += shape[0]*shape[1]
    return var

def bias_variable(name, shape):
    global TOTALWEIGHT
    TOTALWEIGHT += shape[0]
    return tf.get_variable(name,shape,initializer = tf.zeros_initializer())

def conv2d(x, W, padding = 'SAME',strides=[1,1,1,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def batch_norm(input, reuse=False, is_training=args.training):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=is_training, updates_collections=None, scope=tf.get_variable_scope(), reuse = reuse)

def l2_reg_ortho(weight):
    reg = tf.constant(0.)
    Wshape = weight.get_shape()
    if np.size(weight.get_shape().as_list()) == 2:
        cols = int(Wshape[1])
    else:
        cols = int(Wshape[1]*Wshape[2]*Wshape[3])
    rows = int(Wshape[0])
    w1 = tf.reshape(weight,[-1,cols])
    wt = tf.transpose(w1)
    m  = tf.matmul(wt,w1)
    ident = tf.eye(cols,num_columns=cols)

    w_tmp = (m - ident)
    height = w_tmp.get_shape().as_list()[0]
    u = tf.nn.l2_normalize(tf.random_normal([height,1]),dim=0,epsilon=1e-12)
    v = tf.nn.l2_normalize(tf.matmul(tf.transpose(w_tmp), u), dim=0,epsilon=1e-12)
    u = tf.nn.l2_normalize(tf.matmul(w_tmp, v), dim=0,epsilon=1e-12)
    sigma = tf.norm(tf.reshape(tf.keras.backend.dot(tf.transpose(u), tf.matmul(w_tmp, v)),[-1]))
    reg+=sigma**2
    return reg


x = tf.placeholder(tf.float32, [None,imagewidth*imageheight])
y = tf.placeholder(tf.float32, [None,CLASS_NUM])
lr = tf.placeholder(tf.float32)

# forward pass
W_conv1 = weight_variable("W_conv1",[imagewidth*imageheight,CLASS_NUM])
b_conv1 = bias_variable("b_conv1",[CLASS_NUM])
fcout = tf.matmul(x, W_conv1) + b_conv1
# backward pass
back_input = tf.matmul((fcout-b_conv1),tf.transpose(W_conv1))

prediction = tf.reshape(tf.nn.softmax(fcout),[-1,CLASS_NUM])
# calculate the loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
loss = cross_entropy

if args.orthofactor != 0:
    loss = loss + args.orthofactor*l2_reg_ortho(W_conv1)
if args.rrfactor != 0:
    loss = loss + args.rrfactor * tf.reduce_mean(tf.nn.l2_loss(back_input - x))

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.AdamOptimizer(lr).minimize(loss)
# init session
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=maxToKeep)
sess.run(tf.global_variables_initializer())

if args.testnum >= 0 and args.modelnum >=0:
    loadpath = args.basePath+folderpre+"/test_%04d/model_%04d.ckpt"%(args.testnum,args.modelnum)
    saver.restore(sess,loadpath)
    print("Model restored from %s."%(loadpath))

Epochnum = int(np.shape(x_train)[0]/BATCH_SIZE)

def saveModel(test_path, save_no):
    saver.save(sess, test_path+'model_%04d.ckpt'%save_no)
    msg = 'saved Model %04d.'%save_no
    return msg

currenttime = time.time()
testindex = 0
if args.training:
    for i in range(EPOCHS * Epochnum):
        cross_e,_, trainloss = sess.run([cross_entropy , train_step,loss],feed_dict={x: x_train, y: y_train, lr:learningratevalue})
        if i % (Epochnum*100) == 0:
            epochindex = int(i/(Epochnum*100))
            testaccuracy,outputdata= sess.run([accuracy,back_input],feed_dict={x: x_test, y: y_test})
            costtime = time.time()-currenttime
            print("EPOCHS: %d, train loss:%f, testing accuracy:%f, time consuming:%f"%(epochindex,trainloss,testaccuracy,costtime))
            print("cross_e:%f"%cross_e)
            testf.write(str(epochindex)+'\t'+str(testaccuracy)+'\r\n')
            trainf.write(str(epochindex)+'\t'+str(trainloss)+'\r\n')
            timef.write(str(epochindex)+'\t'+str(costtime)+'\r\n')
            if (epochindex+1)%2 == 0:
                print(saveModel(test_path,epochindex))
            # output test image
            outputdata = np.reshape(outputdata,[2,28,28])
            resultpath = test_path +"backwardtest_img/"
            while not os.path.exists(resultpath):
                os.mkdir(resultpath)
            for ind in range(2):
                imageio.imwrite(resultpath + 'test%d_%04d.png'%(ind,testindex),outputdata[ind].astype(np.uint8))
            testindex += 1
            currenttime = time.time()
