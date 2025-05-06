import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem
img = cv2.imread('folhas/img_folha_1.JPG')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Converte para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define faixas de cores
verde_min = np.array([30, 40, 40])
verde_max = np.array([90, 255, 255])
danificado_min = np.array([10, 30, 30])
danificado_max = np.array([30, 255, 255])

# Cria máscaras
mask_verde = cv2.inRange(hsv, verde_min, verde_max)
mask_danificado = cv2.inRange(hsv, danificado_min, danificado_max)

# Operações para limpar as máscaras
kernel = np.ones((3,3), np.uint8)
mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_OPEN, kernel)
mask_danificado = cv2.morphologyEx(mask_danificado, cv2.MORPH_OPEN, kernel)

# Isola as áreas
saudavel = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_verde)
danificado = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_danificado)

# Mostra as imagens
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(saudavel)
plt.title('Parte Saudável')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(danificado)
plt.title('Parte Danificada')
plt.axis('off')

plt.tight_layout()
plt.show()