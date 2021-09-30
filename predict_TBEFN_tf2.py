import os, time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
import numpy as np
import glob
import cv2

checkpoint_dir = './ckpt/'  
input_dir = './input_dir/'
result_dir = './results/'

def out_acti(x):
    return tf.nn.relu(x)-tf.nn.relu(x-1.0)

def denoise_net(input, name): 

    with tf.variable_scope(name):

        conv1_out = slim.conv2d(input, 3, [3, 3], rate=1, activation_fn=None, scope='di_conv1')

        conv2_in = conv1_out
        conv2_out = slim.conv2d(conv2_in, 3, [3, 3], rate=1, activation_fn=None, scope='di_conv2')

        conv3_in = conv1_out + conv2_out
        conv3_out = slim.conv2d(conv3_in, 3, [3, 3], rate=1, activation_fn=None, scope='di_conv3')

        conv4_in = conv3_in + conv3_out
        conv4_out = slim.conv2d(conv4_in, 3, [3, 3], rate=1, activation_fn=None, scope='di_conv4')

        conv5_in = conv4_in + conv4_out
        conv5_out = slim.conv2d(conv5_in, 3, [3, 3], rate=1, activation_fn=None, scope='di_conv5')

        return out_acti(input + conv5_out)

def upsample_and_concat(x1, x2, output_channels, in_channels):  
    with tf.variable_scope("us_vars"):
        pool_size = 2
        deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
        return deconv_output

def simple_unet(input,name): 

    with tf.variable_scope(name):
        conv_1 = slim.conv2d(input, 3, [3, 3], rate=1, activation_fn=None, scope='pp_conv1')
        conv_2 = slim.conv2d(conv_1, 3, [3, 3], rate=1, activation_fn=None, scope='pp_conv2')
        conv_3 = slim.conv2d(conv_2, 3, [3, 3], rate=1, activation_fn=tf.nn.relu,  scope='pp_conv3')
        conv_4 = slim.conv2d(conv_3, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='pp_conv4')

        #fusion
        fu_1 = tf.concat([input, conv_4], 3)

        conv1 = slim.conv2d(fu_1, 16, [3, 3], rate=1, activation_fn=tf.nn.relu,  scope='u_conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')
        conv2 = slim.conv2d(pool1, 32, [3, 3], rate=1, activation_fn=tf.nn.relu,  scope='u_conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
        conv3 = slim.conv2d(pool2, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='u_conv3')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='u_conv4')

        up5 = upsample_and_concat(conv4, conv3, 64, 128)
        conv5 = slim.conv2d(up5, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='u_conv5')

        up6 = upsample_and_concat(conv5, conv2, 32, 64)
        conv6 = slim.conv2d(up6, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='u_conv6')

        up7 = upsample_and_concat(conv6, conv1, 16, 32)
        conv7 = slim.conv2d(up7, 16, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='u_conv7')

        conv8 = slim.conv2d(conv7, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='u_conv8') # modified lk199

        return conv8

def fusion(input_1,input_2,name):
    with tf.variable_scope(name):
        fusion_in = tf.concat([input_1, input_2], 3)
        out_1 = slim.conv2d(fusion_in, 16, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='fusion_1')
        out_2 = slim.conv2d(out_1, 16, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='fusion_2')
        out_3 = slim.conv2d(out_2, 3, [3, 3], rate=1, activation_fn=None, scope='fusion_3')
        return out_3

def atten(input,name):
    with tf.variable_scope(name):
        out_1 = slim.conv2d(input, 16, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='atten_1')
        out_2 = slim.conv2d(out_1, 16, [3, 3], padding='SAME', rate=2, activation_fn=tf.nn.relu, scope='atten_2')
        out_3 = slim.conv2d(out_2, 16, [3, 3], padding='SAME', rate=2, activation_fn=tf.nn.relu, scope='atten_3')
        out_4 = slim.conv2d(out_3, 1, [3, 3], rate=1, activation_fn=None, scope='atten_4')
        return out_acti(out_4)

def buildmodel(sample):

    trans_fun_A_with_1E = simple_unet(sample, name='fun_est_A_with_1E')
    enhanced_1E = out_acti(sample * trans_fun_A_with_1E)

    denoised_in = denoise_net(sample, name='denoise_net')

    trans_fun_B_with_2E = simple_unet(denoised_in, name='fun_est_B_with_2E')
    enhanced_2E = out_acti(denoised_in * trans_fun_B_with_2E)

    atten_map = atten(sample, name='atten')
    fused = atten_map*enhanced_1E + (1-atten_map)*enhanced_2E

    enhanced = fusion(fused, sample, name='fusion')

    return enhanced

# -----------------------------------------#settings and preparations----------
sess = tf.compat.v1.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
uf_out = buildmodel(in_image)
# =------------------------------updates--------------------------------
time_elapsed = 0
with tf.Session() as sess:
    saver = tf.compat.v1.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    # --------------------------------------------------------------------#
    eval_fns = glob.glob(input_dir + '*.*')
    for N in range(len(eval_fns)):

        temp_train = np.array(cv2.imread(eval_fns[N]))
        temp_train = temp_train/255.0
        # ---------------------------------------------------------------------#
        train_data = temp_train.reshape(1, temp_train.shape[0], temp_train.shape[1], temp_train.shape[2])

        st = time.time()
        [out] = sess.run([uf_out], feed_dict={in_image: train_data})
        time_elapsed += time.time() - st
        print('%s' % eval_fns[N])

        [_, name] = os.path.split(eval_fns[N])
        suffix = name[name.find('.') + 1:]
        name = name[:name.find('.')]

        output = np.array(out[0])
        output = output.reshape(output.shape[0], output.shape[1], output.shape[2])
        output = output*255.0
        output_rueslt = np.array(output)

        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        cv2.imwrite(result_dir + name + '_TBEFN.png', output_rueslt)

    print('total processing time: ', time_elapsed)