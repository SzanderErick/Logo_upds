import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import os

TAMANO = 224
BATCH = 32
DIR_DATASET = 'dataset'
DIR_MODELOS = 'modelos'
RUTA_MODELO = os.path.join(DIR_MODELOS, 'logo_upds_mobilenetv2.h5')

modelo = load_model(RUTA_MODELO)

gen_prueba = ImageDataGenerator(rescale=1./255)

prueba_gen = gen_prueba.flow_from_directory(
    os.path.join(DIR_DATASET, 'prueba'),
    target_size=(TAMANO, TAMANO),
    batch_size=BATCH,
    class_mode='categorical',
    classes=['Logo_UPDS', 'No_Logo'],
    shuffle=False
)

perdida, precision = modelo.evaluate(prueba_gen, verbose=1)
print(f"\nPrecision en prueba: {precision:.4f}")
print(f"Perdida en prueba:   {perdida:.4f}")

predicciones = modelo.predict(prueba_gen, verbose=1)
y_pred = np.argmax(predicciones, axis=1)
y_real = prueba_gen.classes

mc = confusion_matrix(y_real, y_pred)
reporte = classification_report(y_real, y_pred, target_names=['Logo_UPDS', 'No_Logo'])
print("\nMatriz de Confusion:")
print(mc)
print("\nReporte de Clasificacion:")
print(reporte)

with open(os.path.join(DIR_MODELOS, 'reporte.txt'), 'w') as f:
    f.write(f"Precision: {precision:.4f}\nPerdida: {perdida:.4f}\n\n")
    f.write("Matriz de Confusion:\n")
    f.write(str(mc) + "\n\n")
    f.write("Reporte:\n" + reporte)

plt.figure(figsize=(6, 5))
plt.imshow(mc, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusion - Logo UPDS')
plt.colorbar()
marcas = np.arange(2)
plt.xticks(marcas, ['Logo_UPDS', 'No_Logo'])
plt.yticks(marcas, ['Logo_UPDS', 'No_Logo'])
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
for i in range(2):
    for j in range(2):
        color = 'white' if mc[i, j] > mc.max() / 2 else 'black'
        plt.text(j, i, str(mc[i, j]), ha='center', va='center', color=color)
plt.tight_layout()
plt.savefig(os.path.join(DIR_MODELOS, 'matriz_confusion.png'))
plt.show()