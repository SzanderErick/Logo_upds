import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

TAMANO = 224
DIR_MODELOS = 'modelos'
RUTA_MODELO = os.path.join(DIR_MODELOS, 'logo_upds_mobilenetv2.h5')
ETIQUETAS = ['Logo_UPDS', 'No_Logo']

modelo = load_model(RUTA_MODELO)

indice_cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(indice_cam)

if not cap.isOpened():
    print(f"Error: No se pudo abrir la camara {indice_cam}.")
    exit()

print("Camara abierta. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (TAMANO, TAMANO))
    img_array = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    prediccion = modelo.predict(img_array, verbose=0)[0]
    clase_idx = np.argmax(prediccion)
    confianza = prediccion[clase_idx]
    clase_detectada = ETIQUETAS[clase_idx]

    if clase_detectada == 'Logo_UPDS':
        texto = f"ES EL LOGO UPDS  ({confianza:.0%})"
        color = (0, 255, 0)
    else:
        texto = f"NO ES EL LOGO UPDS  ({confianza:.0%})"
        color = (0, 0, 255)

    cv2.putText(frame, texto, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    barra_ancho = int(frame.shape[1] * confianza)
    cv2.rectangle(frame, (0, frame.shape[0] - 15), (barra_ancho, frame.shape[0]), color, -1)
    cv2.putText(frame, f"{confianza:.0%}", (barra_ancho + 5, frame.shape[0] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imshow('Deteccion Logo UPDS - MobileNetV2', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()