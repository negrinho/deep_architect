from deep_architect.surrogates.common import SurrogateModel
import numpy as np


class DummySurrogate(SurrogateModel):

    def __init__(self):
        self.val_lst = []

    def eval(self, feats):
        if len(self.val_lst) == 0:
            return 0.0
        else:
            return np.mean(self.val_lst)

    def update(self, val, feats):
        self.val_lst.append(val)