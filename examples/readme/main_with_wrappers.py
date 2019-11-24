import keras
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import RMSprop
import keras.layers as kl

# import deep_architect.helpers.keras_support as hke
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.helpers.common as hco
import deep_architect.helpers.keras_support as hke
import deep_architect.core as co
import deep_architect.visualization as vi
from deep_architect.searchers.common import random_specify

D = hp.Discrete

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def cell(h_units, h_activation, h_rate, h_opt_drop):
    return mo.siso_sequential([
        hke.m_fns["Dense"](h_units, h_activation),
        mo.SISOOptional(lambda: hke.m_fns["Dropout"](h_rate), h_opt_drop)
    ])


def model_search_space():
    h_activation = D(['relu', 'sigmoid'])
    h_rate = D([0.0, 0.25, 0.5])
    h_num_repeats = D([1, 2, 4])
    return mo.siso_sequential([
        mo.SISORepeat(
            lambda: cell(D([256, 512, 1024]), h_activation, D([0.2, 0.5, 0.7]),
                         D([0, 1])), h_num_repeats),
        hke.m_fns["Dense"](num_classes, 'softmax')
    ])


(inputs, outputs) = mo.SearchSpaceFactory(model_search_space).get_search_space()
random_specify(outputs)
inputs_val = kl.Input((784,))
output_name_to_val = hco.simplified_compile_forward(inputs, outputs,
                                                    {"in": inputs_val})
outputs_val = output_name_to_val["out"]

vi.draw_graph(outputs, draw_module_hyperparameter_info=False)
model = Model(inputs=inputs_val, outputs=outputs_val)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])