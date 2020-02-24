import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

import sys

def true_positives(y_true, y_pred):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    return K.sum(K.cast(tf.math.logical_and(y_pred == 1, y_true == 1), dtype=tf.float32))
    # return K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))

def true_negatives(y_true, y_pred):
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    return K.sum(K.cast(tf.math.logical_and(y_pred == 0, y_true == 0), dtype=tf.float32))

def false_positives(y_true, y_pred):
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    return K.sum(K.cast(tf.math.logical_and(y_pred == 1, y_true == 0), dtype=tf.float32))

def false_negatives(y_true, y_pred):
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    return K.sum(K.cast(tf.math.logical_and(y_pred == 0, y_true == 1), dtype=tf.float32))

class SparseCategoricalSpecificity(tf.keras.metrics.Metric):
    
    def __init__(self, num_classes, batch_size,
                 name="sparse_categorical_specificity", **kwargs):
        super(SparseCategoricalSpecificity, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes    

        self.sparse_cat_specificity = self.add_weight(name="sparse_categorical_specificity", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_pred = tf.math.argmax(y_pred, axis=-1)
        
        y_true = K.flatten(y_true)

        true_neg = true_negatives(y_true, y_pred)
        false_pos = false_positives(y_true, y_pred)

        specificity = true_neg / (true_neg + false_pos + 1e-8)
        
        self.sparse_cat_specificity.assign(specificity)

    def result(self):
        
        return self.sparse_cat_specificity

class SparseCategoricalSensitivity(tf.keras.metrics.Metric):
    
    def __init__(self, num_classes, batch_size,
                 name="sparse_categorical_sensitivity", **kwargs):
        super(SparseCategoricalSensitivity, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.sparse_cat_sensitivity = self.add_weight(name="sparse_categorical_sensitivity", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        y_pred = tf.math.argmax(y_pred, axis=-1)
        
        y_true = K.flatten(y_true)

        true_pos = true_positives(y_true, y_pred)
        false_neg = false_negatives(y_true, y_pred)

        sensitivity = true_pos / (false_neg + true_pos + 1e-8)
        
        self.sparse_cat_sensitivity.assign(sensitivity)

    def result(self):

        return self.sparse_cat_sensitivity
    
class SparseCategoricalTruePositives(tf.keras.metrics.Metric):
    
    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(SparseCategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes    

        self.sparse_cat_true_positives = self.add_weight(name="categorical_true_positives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):     
        
        y_pred = tf.math.argmax(y_pred, axis=-1)
        
        y_true = K.flatten(y_true)

        true_pos = true_positives(y_true, y_pred)

        self.sparse_cat_true_positives.assign(true_pos)

    def result(self):

        return self.sparse_cat_true_positives

class CategoricalTruePositives(tf.keras.metrics.Metric):
    
    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes    

        self.cat_true_positives = self.add_weight(name="categorical_true_positives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
                
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        true_pos = true_positives(y_true, y_pred)

        self.cat_true_positives.assign(true_pos)

    def result(self):

        return self.cat_true_positives