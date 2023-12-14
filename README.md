# Code Reference for Tensorflow
## Install
We can install Tensorflow by using pip
```Shell
pip install tensorflow
```
## Import of the tensorflow and keras libraries
Then import the tensorflow library as follows
```Python
import tensorflow as tf
```
In addition we import the Keras API as follows
```Python
import keras
```
Check for the library version
```Python
tf.__version__ # Version check
keras.__version__ # Version check
```
## Tensorflow Basics
### Create Tensor
```Python
# Create a simple tensor
tensor_example = tf.constant([1, 2, 3])
```
## Keras
### Sequantial API
A Sequantial model is appropriate for a plain stack of layers. Each layer has exactly one input and one output tensor. We start by building a classification MLP (Multilayer Perceptron) with two hidden layers
```Python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```


The scheme for a Sequantial model is given as follows
```Python
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
```
### Keras Layers
We summarize different types of layers of the Keras API
```Python
# Core layers
keras.Input()
keras.layers.Dense()
keras.layers.Activation()
keras.layers.Embedding()
keras.layers.Masking()
keras.layers.Lambda()

# Convolution layers
keras.layers.Conv1D()
keras.layers.Conv2D()
keras.layers.Conv3D()

# Pooling layers
keras.layers.MaxPooling1D()

# Recurrent layers
keras.layers.LSTM()

# Preprocessing layers
keras.layers.TextVectorization()

# Regularization layers
keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)

# Reshaping layers
keras.layers.Reshape(target_shape, **kwargs)
keras.layers.Flatten(data_format=None, **kwargs)
```

### Keras tokenizer
We can use the Keras tokenizer instance as follows
```Python
keras.preprocessing.text.Tokenizer(char_level=True)
```
Example
```Python
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(example_text)
```
```Python
```
```Python
```
```Python
```
```Python
```
```Python
```

```Python
```

```Python
```
