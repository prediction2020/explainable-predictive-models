import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback



def roc_auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


class EpochEvaluation(Callback):
    def __init__(self, validation_data=(), training_data=(), test_data=(),batch_size = 8):
        super(Callback, self).__init__()

        self.X_val, self.y_val = validation_data
        self.X_tr, self.y_tr = training_data
        self.X_te, self.y_te = test_data
        self.roc_auc = []
        self.val_roc_auc =[]
        self.test_roc_auc = []
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        y_val_pred = self.model.predict(self.X_val, verbose=0, batch_size=self.batch_size)
        y_tr_pred = self.model.predict(self.X_tr, verbose=0,batch_size = self.batch_size)
        y_te_pred = self.model.predict(self.X_te, verbose=0,batch_size = self.batch_size)
        self.val_roc_auc.append(roc_auc_score(self.y_val, y_val_pred))
        self.roc_auc.append(roc_auc_score(self.y_tr, y_tr_pred))
        self.test_roc_auc.append(roc_auc_score(self.y_te, y_te_pred))
        #logging.info("epoch auc evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
