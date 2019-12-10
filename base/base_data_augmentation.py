class BaseAugmentation(object):
    def __init__(self,data):
        self.data = data

    def augment(self):
        #implement data augmentation
        raise NotImplementedError

     #other supporting methods