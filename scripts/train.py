# scripts/train.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.mesonet_model import build_mesoNet
import os

def train_model(data_dir, model_save_path, target_size=(256, 256), batch_size=32, epochs=20):
    """
    Train the MesoNet model on the prepared dataset.
    """
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    model = build_mesoNet(input_shape=(target_size[0], target_size[1], 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == '__main__':
    DATA_DIR = 'data/processed/'
    MODEL_SAVE_PATH = 'results/trained_model.h5'

    train_model(DATA_DIR, MODEL_SAVE_PATH)
