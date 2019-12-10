from base.base_data_augmentation import BaseAugmentation

class SimpleMnistPreprocess(BaseAugmentation):
    def __init__(self,data):
        super(BaseAugmentation, self).__init__(data)

    def augment(self):
        #implement data preprocessing
        pass