import tensorflow as tf
import numpy as np
from pathlib import Path
import config
import os

def get_class_weights(labels):
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))

def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    
    # Use decode_image which automatically detects BMP, GIF, JPEG, PNG
    # expand_animations=False ensures we get 3D tensors even if a file is a GIF
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    
    # CRITICAL FIX: Explicitly set the shape.
    # Without this, TF sees the shape as <unknown> and 'resize' throws the 
    # "must have either 3 or 4 dimensions" error.
    img.set_shape([None, None, 3])
    
    # --- MAC/METAL FIX: 4D Resize ---
    # Expand to 4D (Batch, H, W, C) -> Resize -> Squeeze back to 3D
    img = tf.expand_dims(img, 0) 
    img = tf.image.resize(img, config.IMG_SIZE)
    img = tf.squeeze(img, 0)
    # --------------------------------
    
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    return img, label

def create_dataset(image_paths, labels, batch_size=config.BATCH_SIZE, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if is_training:
        # Shuffle buffer size based on dataset size
        dataset = dataset.shuffle(buffer_size=max(len(image_paths), 1000))
    
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply batching EXACTLY ONCE
    dataset = dataset.batch(batch_size)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def load_image_paths_and_labels():
    image_paths = []
    labels = []
    
    # Ensure data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
         raise ValueError(f"Data directory 'data' not found in {os.getcwd()}")

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} does not exist.")
            continue
            
        # Search for images (case insensitive usually requires manual globs in python)
        found_files = []
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', 
                   '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP']
        
        for pat in patterns:
            found_files.extend(list(class_dir.glob(pat)))
            
        for img_path in found_files:
            image_paths.append(str(img_path))
            labels.append(class_idx)
            
    if not image_paths:
        raise ValueError("No images found in data directories. Check your 'data' folder.")
        
    return np.array(image_paths), np.array(labels)
