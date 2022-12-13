import tensorflow as tf
backend = tf.keras.backend

class Accuracy(tf.keras.metrics.CategoricalAccuracy):
    def updata_state(self, y_true, y_pred, sample_weight=None):
        return super(Accuracy, self).update_state(y_true, y_pred, sample_weight)
        

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

class TwoClassIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.clip_by_value(y_true, 0, 1)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        return super(TwoClassIoU, self).update_state(y_true, y_pred, sample_weight)
