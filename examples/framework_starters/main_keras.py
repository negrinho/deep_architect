import deep_architect.core as co
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.searchers.random as se
import deep_architect.helpers.common as hco
import deep_architect.visualization as vi

import keras
import keras.layers as kl
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist


class Dense(co.Module):

    def __init__(self, h_units):
        super().__init__(["in0"], ["out0"], {"units": h_units})

    def compile(self):
        self.m = kl.Dense(self.hyperps["units"].val)

    def forward(self):
        self.outputs["out0"].val = self.m(self.inputs["in0"].val)


class Dropout(co.Module):

    def __init__(self, h_drop_rate):
        super().__init__(["in0"], ["out0"], {"drop_rate": h_drop_rate})

    def compile(self):
        self.m = kl.Dropout(self.hyperps["drop_rate"].val)

    def forward(self):
        self.outputs["out0"].val = self.m(self.inputs["in0"].val)


class BatchNormalization(co.Module):

    def __init__(self):
        super().__init__(["in0"], ["out0"], {})

    def compile(self):
        self.m = kl.BatchNormalization()

    def forward(self):
        self.outputs["out0"].val = self.m(self.inputs["in0"].val)


class Nonlinearity(co.Module):

    def __init__(self, h_nonlin_name):
        super().__init__(["in0"], ["out0"], {'nonlin_name': h_nonlin_name})

    def compile(self):
        nonlin_name = self.hyperps["nonlin_name"].val
        self.m = kl.Activation(nonlin_name)

    def forward(self):
        self.outputs["out0"].val = self.m(self.inputs["in0"].val)


def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
             h_drop_rate):
    return mo.sequential([
        Dense(h_num_hidden),
        Nonlinearity(h_nonlin_name),
        mo.Permutation([
            lambda: mo.Optional(lambda: Dropout(h_drop_rate), h_opt_drop),
            lambda: mo.Optional(lambda: BatchNormalization(), h_opt_bn),
        ], h_swap)
    ])


def dnn_net(num_classes):
    h_nonlin_name = hp.Discrete(['relu', 'tanh', 'elu'])
    h_swap = hp.Discrete([0, 1])
    h_opt_drop = hp.Discrete([0, 1])
    h_opt_bn = hp.Discrete([0, 1])
    return mo.sequential([
        mo.Repeat(
            lambda: dnn_cell(hp.Discrete([64, 128, 256, 512, 1024]),
                             h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
                             hp.Discrete([0.25, 0.5, 0.75])),
            hp.Discrete([1, 2, 4])),
        Dense(num_classes)
    ])


class SimpleClassifierEvaluator:

    def __init__(self,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 num_classes,
                 num_training_epochs,
                 batch_size=256,
                 learning_rate=1e-3):

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.num_training_epochs = num_training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def evaluate(self, inputs, outputs):
        keras.backend.clear_session()

        X = kl.Input(self.X_train[0].shape)
        module_eval_seq = co.determine_module_eval_seq(inputs)
        x = co.determine_input_output_cleanup_seq(inputs)
        input_cleanup_seq, output_cleanup_seq = x
        input_name_to_val = {"in0": X}
        output_name_to_val = hco.compile_forward(inputs, outputs,
                                                 input_name_to_val,
                                                 module_eval_seq,
                                                 input_cleanup_seq,
                                                 output_cleanup_seq)

        logits = output_name_to_val["out0"]
        probs = kl.Activation('softmax')(logits)

        model = Model(inputs=[X], outputs=[probs])
        model.compile(optimizer=Adam(lr=self.learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(self.X_train,
                            self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.num_training_epochs,
                            validation_data=(self.X_val, self.y_val))
        results = {'validation_accuracy': history.history['val_accuracy'][-1]}
        return results


def main():
    num_classes = 10
    num_samples = 4
    num_training_epochs = 2
    validation_frac = 0.2
    # NOTE: change to True for graph visualization
    show_graph = False

    # load the data.
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    fn = lambda X: X.reshape((X.shape[0], -1))
    X_train = fn(X_train) / 255.0
    X_test = fn(X_test) / 255.0
    num_train = int((1.0 - validation_frac) * X_train.shape[0])
    X_train, X_val = X_train[:num_train], X_train[num_train:]
    y_train, y_val = y_train[:num_train], y_train[num_train:]

    # define the search and the evalutor
    evaluator = SimpleClassifierEvaluator(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes,
        num_training_epochs=num_training_epochs)
    search_space_fn = lambda: dnn_net(num_classes)
    searcher = se.RandomSearcher(search_space_fn)

    for i in range(num_samples):
        (inputs, outputs, hyperp_value_lst,
         searcher_eval_token) = searcher.sample()
        if show_graph:
            # try setting draw_module_hyperparameter_info=False and
            # draw_hyperparameters=True for a different visualization.
            vi.draw_graph(outputs,
                          draw_module_hyperparameter_info=False,
                          draw_hyperparameters=True)
        results = evaluator.evaluate(inputs, outputs)
        # updating the searcher. no-op for the random searcher.
        searcher.update(results['validation_accuracy'], searcher_eval_token)


if __name__ == "__main__":
    main()