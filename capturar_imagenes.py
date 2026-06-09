import cv2
import os
import random
from datetime import datetime

TAMANO = 224
DIR_DATASET = 'dataset'
CLASES = {
    'l': ('Logo_UPDS', (0, 255, 0)),
    'n': ('No_Logo', (0, 0, 255))
}
SPLIT = {'train': 0.7, 'validacion': 0.2, 'prueba': 0.1}

for clase in ['Logo_UPDS', 'No_Logo']:
    for carpeta in SPLIT:
        os.makedirs(os.path.join(DIR_DATASET, carpeta, clase), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la camara.")
    exit()

print("=" * 50)
print("  CAPTURADOR - LOGO UPDS  ")
print("=" * 50)
print("  l = capturar LOGO UPDS")
print("  n = capturar NO LOGO")
print("  q = salir")
print("=" * 50)

def contar_imagenes(clase):
    total = 0
    for c in SPLIT:
        ruta = os.path.join(DIR_DATASET, c, clase)
        total += len(os.listdir(ruta))
    return total

while True:
    ret, frame = cap.read()
    if not ret:
        break

    logo_cnt = contar_imagenes('Logo_UPDS')
    nologo_cnt = contar_imagenes('No_Logo')

    info = f"LOGO: {logo_cnt}  NO LOGO: {nologo_cnt}  [l=logo  n=no_logo  q=salir]"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    h, w = frame.shape[:2]
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "ENCUADRE AQUI", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Capturar Imagenes - UPDS', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord('l'), ord('n')]:
        nombre_clase, color = CLASES[chr(key)]
        destino = random.choices(list(SPLIT.keys()), weights=list(SPLIT.values()))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        nombre_archivo = f"{nombre_clase}_{ts}.png"
        ruta_guardado = os.path.join(DIR_DATASET, destino, nombre_clase, nombre_archivo)
        roi = frame[y1:y2, x1:x2]
        roi_resize = cv2.resize(roi, (TAMANO, TAMANO))
        cv2.imwrite(ruta_guardado, roi_resize)
        print(f"[{destino.upper()}] Guardado: {ruta_guardado}")

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal -> Logo_UPDS: {contar_imagenes('Logo_UPDS')}, No_Logo: {contar_imagenes('No_Logo')}")
print("Ejecuta: python entrenar.py")