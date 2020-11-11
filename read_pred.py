import numpy as np
import tensorflow as tf
import inputs

pred_path = './data/UrbanSound8K/pred'
pred_sound_ph, pred_label_ph = inputs.load_data([pred_path])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

snd, label = sess.run([pred_sound_ph, pred_label_ph])
print(snd.shape, snd.dtype, label)
print(np.max(snd), np.min(snd))