from utils import *

from tensorflow.keras import models, layers, activations, losses, metrics, optimizers

feature_extractor = models.Sequential([
    layers.Conv2D(9,(7,7), activation="relu", input_shape = (28,28,1), data_format="channels_last"),     # layer is functioning as block
    layers.Conv2D(9,(7,7), activation="relu"),     # Calculating 7 as 28/4
    layers.Conv2D(9,(7,7), activation="relu"),
    layers.Conv2D(9,(7,7), activation="relu"),
])

keras_model = models.Sequential([
    feature_extractor,
    # compress to 10 classes. Embedding of all previous in 10 groups
    layers.Conv2D(10,(1,1), activation="relu", input_shape=(4,4,39)),  
    layers.Conv2D(10*3, (3,3), activation="relu"),  # down sample to 1*1, 3 major features per class
    layers.Conv2D(10*15, (1,1), activation="relu"), # assuming 15 features per class
    layers.Conv2D(10,(1,1), activation="relu"),  # compress to 10 classes again
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax")
])

keras_model.build((28,28,1))
logger.debug(keras_model.summary())

# %% [code]

keras_model.compile(
    loss = "categorical_crossentropy",
    optimizer = "sgd",
    metrics = ["accuracy", ],
)

keras_model.fit(x_train,y_train, epochs = 1)
keras_model.evaluate(x_val,y_val)

# %% [code]
y_predicted_probs = keras_model.predict(x_client)

save_predictions_csv(y_predicted_probs)
