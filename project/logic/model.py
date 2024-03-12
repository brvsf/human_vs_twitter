# Final model python file

from preprocess import X_pad, y_train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# Set parameters
top_3 = TopKCategoricalAccuracy(
    k=3, name='top_3_accuracy', dtype=None
)

input_shape = X_pad.shape[1:]# Maximum feature length
loss = 'categorical_crossentropy' # Loss function
optimizer= 'adam'  # Optimizer
metrics= ['accuracy', top_3]  # Defining metric to accuracy
es = EarlyStopping(patience=3) # Defining Early stopping

def init_model():
    """Function to create model architecture"""
    model = Sequential() # Instanciate the model
    model.add(LSTM(neurons=64, activation='tanh', input_shape = input_shape)) # Input layer
    model.add(Dense(neurons=32, activation='tanh')) # Hidden layer
    model.add(Dense(neurons=28, activation='softmax')) # Output layer

    return model

def compile_model(model):
    """Function to compile the model"""
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )
    return model

model = init_model()
model = compile_model(model)

history = model.fit(
    X_pad,
    y_train,
    batch_size=264,
    epochs=500,
    validation_split=0.2,
    callbacks=[es]
)
