import os
import tensorflow as tf
import numpy as np  # Numpy import zaroori hai

# 🔥 TURN OFF MOST KERAS LOGGING
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# =========================
# CONFIG
# =========================
BASE_DIR = r"D:\plant_disease_prediction\dataset\tomato_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# =========================
# SAFE LOGS CALLBACK
# =========================
class SafeFloatCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        clean_logs = {}
        for key, value in logs.items():
            # TensorFlow tensors
            if tf.is_tensor(value):
                v = value.numpy()
                if v.size == 1:
                    clean_logs[key] = float(v.reshape(()))
                else:
                    clean_logs[key] = v.tolist()

            # NumPy arrays
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    clean_logs[key] = float(value.reshape(()))
                else:
                    clean_logs[key] = value.tolist()

            # Python scalars
            elif isinstance(value, (float, int)):
                clean_logs[key] = value

            # Fallback
            else:
                clean_logs[key] = str(value)

        logs.clear()
        logs.update(clean_logs)

# =========================
# DATA
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_generator.num_classes

# =========================
# MODEL (EfficientNetV2B0)
# =========================
base_model = EfficientNetV2B0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # transfer learning freeze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# CALLBACKS
# =========================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_tomato_transfer_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=0
)

safe_float_cb = SafeFloatCallback()

# =========================
# TRAIN
# =========================
print("\n🚀 Training started (EfficientNetV2B0 + GPU, SafeFloatCallback active)...\n")

try:
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[safe_float_cb, early_stopping, checkpoint],
        verbose=1  # 1 = per-epoch logs; 0 chahiye to change kar dena
    )

    # Final model save
    model.save("final_tomato_transfer_model.keras")
    print("\n✅ Training finished and model saved successfully")

except Exception as e:
    print(f"\n❌ Error during training: {e}")
