import cv2
import matplotlib.pyplot as plt 
import numpy as np

# Carrega a imagem
img = cv2.imread('/home/vinicius/Python/PDI/circulos/circulos_1.png')
copia = img.copy()

# Converte para tons de cinza
img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecta círculos com Hough
circulos = cv2.HoughCircles(
    img_cinza,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=80,       # Maior distância entre centros
    param1=100,       # Threshold alto para borda (Canny)
    param2=50,        # Threshold mais alto para o centro — menos falsos
    minRadius=40,     # Ajuste com base no tamanho real das fichas
    maxRadius=60
)



# Se houver círculos, desenha
if circulos is not None:
    circulos = np.round(circulos[0, :]).astype("int")

    for (x, y, r) in circulos:
        cv2.circle(copia, (x, y), r, (0, 255, 0), 4)  # Círculo
        cv2.rectangle(copia, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Centro

# Exibe imagens
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(copia, cv2.COLOR_BGR2RGB))
qtd = len(circulos) if circulos is not None else 0
plt.title(f"Círculos Detectados: {qtd}")
plt.axis('off')


plt.tight_layout()
plt.show()
