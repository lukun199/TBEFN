from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import cv2

sess = tf.Session()
with gfile.FastGFile('../Pb_File/TBEFN.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

input_image = sess.graph.get_tensor_by_name('Placeholder:0')
output_image = sess.graph.get_tensor_by_name('fusion/fusion_3/BiasAdd:0')

inp = cv2.imread('../input_dir/03.bmp')/255.
inp = inp.reshape(1, inp.shape[0], inp.shape[1], inp.shape[2])

res = sess.run(output_image,  feed_dict={input_image:inp})
res = np.array(res[0])
cv2.imwrite('../Pb_File/PbOutTest.png', res*255.)
print('\n\n [*]------Pb file test OK!')
