import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras


def load_images(path: str, X_shape: tuple) -> np.ndarray:
    images = []  
    
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return np.array([])
    
    for filename in os.listdir(path):
        image_path = os.path.join(path, filename)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        image = cv2.resize(image, X_shape[:2]) #

        
        
        # Normalize to [0,1]
        image = image / 255.0
        
        images.append(image)
    
    if len(images) == 0:
        print(f"No valid images found in {path}")
        return np.array([])
    
    return np.array(images)


def load_data(path: str, X_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    N = load_images(path + "/N", X_shape)
    P = load_images(path + "/P", X_shape)
    return N, P


def augment_data(
    X: tuple[np.ndarray, np.ndarray], n: int
) -> tuple[np.ndarray, np.ndarray]:
    X_N, X_P = X

    augment = keras.Sequential(
        [
            # Random horizontal flip
            layers.RandomFlip("horizontal"),
            # Random rotation (up to 10 degrees)
            layers.RandomRotation(0.1),
            # Random zoom (10% zoom range)
            layers.RandomZoom(0.1),
            # Random translation (10% of image size)
            layers.RandomTranslation(0.1, 0.1),
            # Random brightness adjustment
            layers.RandomBrightness(0.2),
            # Random contrast adjustment
            layers.RandomContrast(0.2),
            # add noise
            layers.GaussianNoise(0.1),
        ]
    )

    X_N_aug_list = []
    X_P_aug_list = []

    # Apply augmentation n times
    for _ in range(n):
        # Augment the entire batch of X_N and X_P
        X_N_aug_list.append(augment(X_N, training=True))
        X_P_aug_list.append(augment(X_P, training=True))

    # Convert lists to NumPy arrays and concatenate along the batch axis
    X_N_aug = np.concatenate(X_N_aug_list, axis=0)
    X_P_aug = np.concatenate(X_P_aug_list, axis=0)

    
    return X_N_aug, X_P_aug

   


def generate_image_pairs(X: tuple[np.ndarray, np.ndarray])->tuple[np.ndarray, np.ndarray]:
    n_N = X[0].shape[0]
    n_P = X[1].shape[0]

    X_P_pairs = np.array(
        [(X[1][i], X[1][j]) for j in range(n_P) for i in range(n_P) if i != j]
    )
    X_N_pairs = np.array([(X[0][i], X[1][j]) for j in range(n_P) for i in range(n_N)])

    ### generate the labels ###
    y_P: np.ndarray = np.ones(X_P_pairs.shape[0])
    y_N: np.ndarray = np.zeros(X_N_pairs.shape[0])

    ds_X: np.ndarray = np.vstack([X_P_pairs, X_N_pairs])
    ds_Y: np.ndarray = np.hstack([y_P, y_N]) 
   

    return ds_X, ds_Y


def data_pipeline(
    path: str, X_shape: tuple, n: int, train_ratio: float, P_to_N_ratio: float
)->tuple[np.ndarray, np.ndarray]:
    X = load_data(path, X_shape)

    X = augment_data(X, n)

    DS = generate_image_pairs(X)

    ds_X, ds_y = DS

    train_test_split = int(train_ratio * ds_X.shape[0])

    while True:
        # shuffle the data
        idxs: np.ndarray = np.arange(ds_X.shape[0])
        np.random.shuffle(idxs)
        ds_X, ds_y = ds_X[idxs], ds_y[idxs]

        X_train, X_test = ds_X[:train_test_split], ds_X[train_test_split:]
        y_train, y_test = ds_y[:train_test_split], ds_y[train_test_split:]

        n_positive_train = np.sum(y_train == 1)
        n_negative_train = np.sum(y_train == 0)

        if n_positive_train / n_negative_train >= P_to_N_ratio:
            break

    return (X_train, y_train), (X_test, y_test)


def main() -> int:

    path = "dbs\comparator_db\\raw"
    X_shape = (224, 224, 3)
    train_ratio = 0.8
    n = 4
    P_to_N_ratio = 0.4

    
    (X_train, y_train), (X_test, y_test) = data_pipeline(
        path, X_shape, n, train_ratio, P_to_N_ratio
    )

    return 0


if __name__ == "__main__":
    main()
