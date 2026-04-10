import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import config
import data_preprocessing
import os
from sklearn.model_selection import StratifiedKFold

def create_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3)
    )
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate_kfold(image_paths, labels, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    for train_index, val_index in skf.split(image_paths, labels):
        print(f'Training fold {fold_no}...')
        X_train, X_val = image_paths[train_index], image_paths[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        train_dataset = data_preprocessing.create_dataset(X_train, y_train, is_training=True)
        val_dataset = data_preprocessing.create_dataset(X_val, y_val, is_training=False)

        class_weights = data_preprocessing.get_class_weights(y_train)
        print("Class weights:", class_weights)

        model = create_model()
        model = compile_model(model)
        model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.PATIENCE, restore_best_weights=True),
            ModelCheckpoint(filepath=str(config.MODEL_DIR / f'best_model_fold_{fold_no}.h5'),
                            monitor='val_accuracy', save_best_only=True)
        ]
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Fine-tuning
        for layer in model.layers:
            if isinstance(layer, tf.keras.models.Model):
                for i, base_layer in enumerate(layer.layers):
                    base_layer.trainable = i >= len(layer.layers) - 20
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        fine_tune_callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.PATIENCE, restore_best_weights=True),
            ModelCheckpoint(filepath=str(config.MODEL_DIR / f'fine_tuned_model_fold_{fold_no}.h5'),
                            monitor='val_accuracy', save_best_only=True)
        ]
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            callbacks=fine_tune_callbacks,
            class_weight=class_weights
        )

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(fine_tune_history.history['accuracy'], label='Training Accuracy')
        plt.plot(fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Fine-tuning Accuracy Fold {fold_no}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(fine_tune_history.history['loss'], label='Training Loss')
        plt.plot(fine_tune_history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fine-tuning Loss Fold {fold_no}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(config.RESULTS_DIR / f'fine_tuning_history_fold_{fold_no}.png'))
        plt.close()

        fold_no += 1

    print("Cross-validation training complete.")

def main():
    image_paths, labels = data_preprocessing.load_image_paths_and_labels()
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    train_and_evaluate_kfold(image_paths, labels, n_splits=5)

if __name__ == "__main__":
    main()
