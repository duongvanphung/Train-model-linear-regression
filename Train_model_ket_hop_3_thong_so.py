from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import numpy as np

"""IMPORT DỮ LIỆU TỪ FILE .CSV"""

outputs = []
fvecs = []
with open("./data.csv", "r") as ins:
  for line in ins:
    row = line.split(",")
    outputs.append(int(row[0]))
    fvecs.append([int(x) for x in row[1:]])
X = np.array(fvecs)
Y = np.array(outputs)
Y = Y.reshape(len(Y), 1)
X.shape

"""TRAINING MODEL"""

model = Sequential()
model.add(Dense(32, input_dim=3, activation='tanh'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer ='adam',metrics =['accuracy'])
model.fit(X, Y, batch_size=32, nb_epoch=1000)

"""LƯU MODEL"""
tf.train.write_graph(K.get_session().graph_def, './', \
    'name_graph.pbtxt')

tf.train.Saver().save(K.get_session(), './name.chkp')

freeze_graph.freeze_graph('./name_graph.pbtxt', None, \
    False, './name.chkp', "dense_3/Sigmoid", \
    "save/restore_all", "save/Const:0", \
    './frozen_name.pb', True, "")

input_graph_def = tf.GraphDef()
with tf.gfile.Open('./frozen_name.pb', "rb") as f:
    input_graph_def.ParseFromString(f.read())

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, ["dense_1_input"], ["dense_3/Sigmoid"],
        tf.float32.as_datatype_enum)

with tf.gfile.FastGFile('./tensorflow_lite_name.pb', "wb") as f:
    f.write(output_graph_def.SerializeToString())
model.summary()

"""TÍNH ĐỘ CHÍNH XÁC CỦA MODEL"""

_, accuracy = model.evaluate(X, Y, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

"""SHOW TESTING TRÊN TẬP TRAIN"""

predictions = model.predict_classes(X)
for i in range(X.shape[0]):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))

"""TEST 1 GIÁ TRỊ"""

Xnew = np.array([[96.23, 2.32, 23.54]])
#ynew = model.predict_classes(Xnew)
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
