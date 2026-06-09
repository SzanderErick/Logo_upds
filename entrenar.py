import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

TAMANO = 224
BATCH = 32
EPOCAS = 20
LR = 0.0005
NUM_CLASES = 2
DIR_DATASET = 'dataset'
DIR_MODELOS = 'modelos'

os.makedirs(DIR_MODELOS, exist_ok=True)

generador_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    shear_range=0.1
)

generador_val = ImageDataGenerator(rescale=1./255)

train_gen = generador_train.flow_from_directory(
    os.path.join(DIR_DATASET, 'train'),
    target_size=(TAMANO, TAMANO),
    batch_size=BATCH,
    class_mode='categorical',
    classes=['Logo_UPDS', 'No_Logo'],
    shuffle=True
)

val_gen = generador_val.flow_from_directory(
    os.path.join(DIR_DATASET, 'validacion'),
    target_size=(TAMANO, TAMANO),
    batch_size=BATCH,
    class_mode='categorical',
    classes=['Logo_UPDS', 'No_Logo'],
    shuffle=False
)

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(TAMANO, TAMANO, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
salida = Dense(NUM_CLASES, activation='softmax')(x)

modelo = Model(inputs=base.input, outputs=salida)

modelo.compile(
    optimizer=Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    os.path.join(DIR_MODELOS, 'logo_upds_mobilenetv2.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

parada_temprana = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

historial = modelo.fit(
    train_gen,
    steps_per_epoch=max(1, train_gen.samples // BATCH),
    validation_data=val_gen,
    validation_steps=max(1, val_gen.samples // BATCH),
    epochs=EPOCAS,
    callbacks=[checkpoint, parada_temprana],
    verbose=1
)

modelo.save(os.path.join(DIR_MODELOS, 'logo_upds_mobilenetv2.h5'))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Train Acc')
plt.plot(historial.history['val_accuracy'], label='Val Acc')
plt.title('Precisión por época')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Train Loss')
plt.plot(historial.history['val_loss'], label='Val Loss')
plt.title('Pérdida por época')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(DIR_MODELOS, 'curvas_entrenamiento.png'))
plt.show()

print(f"\nAccuracy Train:      {historial.history['accuracy'][-1]:.4f}")
print(f"Accuracy Validacion: {historial.history['val_accuracy'][-1]:.4f}")
print(f"Loss Train:          {historial.history['loss'][-1]:.4f}")
print(f"Loss Validacion:     {historial.history['val_loss'][-1]:.4f}")