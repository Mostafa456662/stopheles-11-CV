import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras as k

#Local Imports
from Siamese_helper import PredictionCallback
from data.siamese_data_pipeline import data_pipeline as dp





class SiameseModel(k.Model):
    def __init__(self, img_shape: tuple[int, ...], embedding_model: k.Model):
        super(SiameseModel, self).__init__()
        self.f = embedding_model
        self.f.trainable = False  
        self.img_shape = img_shape
    
    
        self.g = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation="relu", kernel_regularizer="l2"),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(1, activation="sigmoid", kernel_regularizer="l2"),
        ])

    def call(self, X: np.ndarray) -> np.ndarray:
        X1, X2 = X[:, 0], X[:, 1]
        f1, f2 = self.f(X1), self.f(X2)
        f = tf.concat([f1, f2], axis=1)
        g = self.g(f)
        return g

    def load_model(self, path: str):
        self.load_weights(path)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get embeddings for input images"""
        return self.f(X)

    def predict_similarity(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Predict similarity between two sets of images"""
        f1, f2 = self.f(X1), self.f(X2)
        f = tf.concat([f1, f2], axis=1)
        return self.g(f)


def load_embedding_MobileNetV2(img_shape: tuple[int, ...]) -> k.Model:
    # load the embedding model
    model = k.applications.MobileNetV2(
        input_shape=img_shape, include_top=False, weights="imagenet"
    )

    return model


def create_model(img_shape: tuple[int, ...] = (224, 224, 3)) -> SiameseModel:
    """Create and return a Siamese model"""
    embedding_model = load_embedding_MobileNetV2(img_shape)
    model = SiameseModel(img_shape, embedding_model)
    return model


def compile_model(model: SiameseModel, lr_schedule) -> SiameseModel:
    """Compile the Siamese model with appropriate loss and metrics"""
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=lr_schedule),
        loss="binary_crossentropy",
        metrics=[
        k.metrics.BinaryAccuracy(), 
        # k.metrics.Precision(), 
        # k.metrics.Recall()
    ]
    )
    return model


def train(
    model: SiameseModel,
    X_train,
    y_train,
    epochs: int,
    val_data=None,
    batch_size: int = 32,
    callbacks=None,
    show_predictions=True,
) -> int:
    """
    Train the Siamese model
    Args:
        model: Compiled Siamese model
        train_data: Training data
        val_data: Validation data
        epochs: Number of training epochs
        batch_size: Batch size for training
        callbacks: List of Keras callbacks
    """

    if callbacks is None:
        callbacks = [
            k.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True, monitor="val_loss"
            ),
            k.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
            k.callbacks.ModelCheckpoint(
                "./src/models/model.keras", save_best_only=True, 
            ),
        ]
    if show_predictions:
        pred_callback = PredictionCallback(X_train, y_train, sample_size=15)
        callbacks.append(pred_callback)
    class_weight = {
            0: 1,     
            1: 5     
    }

    model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        class_weight = class_weight,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return 0


def evaluate_model(model: SiameseModel, X_test, y_test):
    """Evaluate the model on test data"""
    results = model.evaluate(X_test, y_test, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    return metrics


if __name__ == "__main__":

    path = "dbs\comparator_db\\raw"
    X_shape = (224, 224, 3)
    train_ratio = 0.8
    n = 4
    P_to_N_ratio = 0.2

    (X_train, y_train), (X_test, y_test) = dp(
        path, X_shape, n, train_ratio, P_to_N_ratio
    )

    # Create model
    img_shape = (224, 224, 3)
    model = create_model(img_shape)
    epochs = 10
    first_phase_epochs = int(0.3 * epochs)
    second_phase_epochs = int(0.7 * epochs)

    model = compile_model(
        model,
        lr_schedule=k.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9
        ),
    )

    # Step 1: Train only the siamese head with frozen embeddings
    print("Step 1: Training siamese head with frozen embeddings...")

    train(model, X_train, y_train, epochs=first_phase_epochs)
    results = model.evaluate(X_test, y_test)
    print(f"The first training phase results: {results}")
    # Step 2: Unfreeze and train the full model end-to-end
    model.f.trainable = True

    model = compile_model(
        model,
        lr_schedule=k.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5, decay_steps=10000, decay_rate=0.9
        ),
    )
    
    train(
        model,
        X_train,
        y_train,
        epochs=second_phase_epochs,  # Fine-tune everything together
    )

    # Evaluate

    results = model.evaluate(X_test, y_test)
    print(f"The second training phase results: {results}")
