from search_space import dnn_net
from deep_architect.searchers.common import specify
import deep_architect.utils as ut 
import deep_architect.core as co 
import argparse
import keras 
import numpy as np 

def get_search_space(num_classes):
    def fn(): 
        co.Scope.reset_default_scope()
        inputs, outputs = dnn_net(num_classes)
        return inputs, outputs, {}
    return fn
    
def load_data(path='datasets/mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# Evaluator 
class SimpleClassifierEvaluator:

    def __init__(self, train_dataset, num_classes, max_num_training_epochs=10, 
                batch_size=256, learning_rate=1e-3):

        self.train_dataset = train_dataset
        self.num_classes = num_classes
        self.max_num_training_epochs = max_num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.val_split = 0.1 # 10% of dataset for validation

    def evaluate(self, inputs, outputs, hs):
        keras.backend.clear_session() 

        (x_train, y_train) = self.train_dataset

        X = keras.layers.Input(x_train[0].shape)
        co.forward({inputs['In'] : X})
        logits = outputs['Out'].val
        probs = keras.layers.Softmax()(logits)
        model = keras.models.Model(inputs=[inputs['In'].val], outputs=[probs])
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
        model.summary() 
        history = model.fit(x_train, y_train, 
                batch_size=self.batch_size, 
                epochs=self.max_num_training_epochs, 
                validation_split=self.val_split)

        results = {'val_accuracy': history.history['val_acc'][-1]}
        return results 

def main(args): 

    num_classes = 10 

    # load and normalize data 
    (x_train, y_train),(x_test, y_test) = load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # defining evaluator, and logger
    evaluator = SimpleClassifierEvaluator((x_train, y_train), num_classes, 
                                        max_num_training_epochs=5)
    inputs, outputs, hs = get_search_space(num_classes)() 
    h_values = ut.read_jsonfile(args.config) 
    specify(outputs.values(), hs, h_values["hyperp_value_lst"]) # hs is "extra" hyperparameters
    results = evaluator.evaluate(inputs, outputs, hs) 
    ut.write_jsonfile(results, args.result_fp)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', 
                        type=str, 
                        default='', 
                        required=True,
                        help='Config file that contains the hyperparameters values determined by DeepArchitect.')
    parser.add_argument('--result_fp', 
                        type=str, 
                        default='', 
                        required=True,
                        help='Filepath to store result of each worker.')

    args = parser.parse_args()
    main(args)