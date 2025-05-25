import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory


data_dir = "ENTER"
epochs = 10
batch_size = 32
lr = 1e-4
val_split = 0.2

train_data = image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    )

val_data  = image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    )

class_names = train_data.class_names
num_classes = len(class_names)

data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.map(lambda x, y: (data_augmentation(x), y)).prefetch(AUTOTUNE)
val_data   = val_data.prefetch(AUTOTUNE)

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
base_model.trainable = False  

inputs = layers.Input(shape=(224,224,3))
x = tf.keras.applications.resnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "Wound_model.h5", save_best_only=True, monitor='val_accuracy'
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3, restore_best_weights=True
)

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print("Training complete. Best model saved to", "Wound_model.h5")

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr/10),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    train_data,
    epochs=5,
    validation_data=val_data,
    callbacks=[checkpoint_cb]
)
print("Fine-tuning complete. Updated model saved to", "Wound_model.h5")
