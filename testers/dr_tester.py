from base.base_tester import BaseTester
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix
import numpy as np

class DRModelTester(BaseTester):
    def __init__(self, model, data, config):
        super(DRModelTester, self).__init__(model, data, config)
        
    def test(self):
        predictions = np.asarray([], dtype=int)
        ground_truth = np.asarray([], dtype=int)
        for x, y in self.data:
            predictions = np.append(predictions, np.argmax(self.model.predict(x), axis=-1))
            ground_truth = np.append(ground_truth, np.argmax(y, axis=-1))
        con_matrix = confusion_matrix(ground_truth, predictions)
        print('Confusion Matrix: {}'.format(con_matrix))
        np.save(os.path.join(self.config.results.performance_dir, 'confusion_matrix.npy'), con_matrix)
