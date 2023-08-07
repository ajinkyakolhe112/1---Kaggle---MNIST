import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

keras_model = keras.models.Sequential([
    keras.layers.Input(shape=(28,28,1)), # channels last assumption. 
    # A shape tuple (integers), not including the batch size. Not pythonic to make batch explicit
    keras.layers.Flatten(),
    keras.layers.Dense(4,activation=keras.activations.relu), # relu beautiful. standard naming
    keras.layers.Dense(10,activation=keras.activations.softmax)
])

keras_model.build(input_shape=(1,28,28,1))

def test_single_example():
    random_num_gen  = np.random.default_rng()
    single_example  = random_num_gen.normal(size=(1,28,28,1))
    
    test_output     = keras_model(single_example)
    test_output

    assert np.sum(test_output,axis=1) == 1
    assert tf.reduce_sum(test_output, axis=1).numpy() == 1
    # AttributeError: 'tensorflow...ops.EagerTensor' object has no attribute '.item()' but has '.numpy()'
