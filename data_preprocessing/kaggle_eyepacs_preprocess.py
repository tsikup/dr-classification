from base.base_data_preprocess import BasePreprocess

class SimpleMnistPreprocess(BasePreprocess):
    def __init__(self,data):
        super(BasePreprocess, self).__init__(data)

    def preprocess(self):
        #implement data preprocessing
        pass