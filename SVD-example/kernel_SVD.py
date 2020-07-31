import numpy as np
import tensorflow as tf
import os
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--testnum", type=int, default=1, help="test index")
parser.add_argument("--modelnum", type=int, default=49, help="model index")
parser.add_argument("--foldername", type=str, default="RR", help="test name")
args = parser.parse_args()
name = args.foldername
testnum=args.testnum
modelnum=args.modelnum
with tf.Session() as sess:
    model_saver = tf.train.import_meta_graph('%s/test_%04d/model_%04d.ckpt.meta'%(name,testnum,modelnum))
    model_saver.restore(sess, "%s/test_%04d/model_%04d.ckpt"%(name,testnum,modelnum))
    globalvar = tf.global_variables()
    for v in globalvar:
        if (v.name).find("W_conv1:0")>=0:
            print("kernel name:%s"%v.name)
            k = sess.run(v)
            print("kernel shape:%s"%str(np.shape(k)))

datawidth = 28
dataheight = 28

vh,s,u = np.linalg.svd(k,full_matrices=True)
print("shape of u:%s"%str(np.shape(u)))
print("shape of s:%s"%str(np.shape(s)))
print("shape of vh:%s"%str(np.shape(vh)))

v=np.reshape(vh,[datawidth*dataheight,datawidth,dataheight])
u=np.transpose(u)
savepath = name+"_SVD_%04d_%04d/"%(testnum,modelnum)
if not os.path.exists(savepath):
    os.mkdir(savepath)
for ind in range(10):   # only check 10 images
    vcolumn = v[ind,:]
    imageio.imwrite(savepath + "v_%04d.png"%(ind),vcolumn.astype(np.uint8))
# for ind in range(np.shape(u)[0]):
#     ucolumn = u[ind:ind+1,:]
#     imageio.imwrite(savepath + "u_%04d.png"%(ind),ucolumn.astype(np.uint8))
