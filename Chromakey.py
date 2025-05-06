import cv2
import matplotlib.pyplot as plt 
import numpy as np

imgverde = cv2.imread('chromakey/img_fundo_verde_1.jpg')
# Converte a s cores para LAB para melhor segmentação
img_lab = cv2.cvtColor(imgverde, cv2.COLOR_BGR2LAB)

# Extrai o canal 'a' (verde-magenta)
canal_a = img_lab[:, :, 1]

# Aplica limiarização com Otsu 
_, thresh = cv2.threshold(canal_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Aplica a máscara à imagem original
mascara = cv2.bitwise_and(imgverde, imgverde, mask=thresh)

# Define pixels fora da máscara como brancos
m1 = mascara.copy()
m1[thresh == 0] = (255, 255, 255)

# Converte as cores para LAB para melhor segmentação
mlab = cv2.cvtColor(mascara, cv2.COLOR_BGR2LAB)

# Normaliza o canal 'a'
dst = cv2.normalize(mlab[:, :, 1], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Aplica outro threshold para reduzir ruído 
_, dst_thresh = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY_INV)

# Remove ruído alterando o canal 'a'
mlab2 = mlab.copy()
mlab2[:, :, 1][dst_thresh == 255] = 127

# Converte de volta para BGR
img_sem_fundo = cv2.cvtColor(mlab2, cv2.COLOR_LAB2BGR)

# Define fundo como branco
img_sem_fundo[thresh == 0] = (255, 255, 255)

background = cv2.imread('chromakey/background_1.png')
# Redimensiona o fundo para combinar com a imagem original
background = cv2.resize(background, (imgverde.shape[1], imgverde.shape[0]))

# Cria a imagem final
mask_inv = cv2.bitwise_not(thresh)
fundo_area = cv2.bitwise_and(background, background, mask=mask_inv)
frente_area = cv2.bitwise_and(img_sem_fundo, img_sem_fundo, mask=thresh)
final_img = cv2.add(fundo_area, frente_area)

plt.figure(figsize=(16, 8))

# 1. Imagem original
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(imgverde, cv2.COLOR_BGR2RGB))
plt.title("1. Imagem Original")
plt.axis('off')

# 2. Canal 'a' do espaço LAB
plt.subplot(2, 4, 2)
plt.imshow(canal_a, cmap='gray')
plt.title("2. Canal A (LAB)")
plt.axis('off')

# 3. Máscara com Otsu
plt.subplot(2, 4, 3)
plt.imshow(thresh, cmap='gray')
plt.title("3. Máscara Otsu")
plt.axis('off')

# 4. Aplicação da máscara na imagem original
plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(mascara, cv2.COLOR_BGR2RGB))
plt.title("4. Máscara Aplicada")
plt.axis('off')

# 5. Canal A normalizado
plt.subplot(2, 4, 5)
plt.imshow(dst, cmap='gray')
plt.title("5. Canal A Normalizado")
plt.axis('off')

# 6. Máscara para remoção de ruído
plt.subplot(2, 4, 6)
plt.imshow(dst_thresh, cmap='gray')
plt.title("6. Máscara de Ruído")
plt.axis('off')

# 7. Imagem com fundo branco (sem ruído)
plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(img_sem_fundo, cv2.COLOR_BGR2RGB))
plt.title("7. Fundo Removido")
plt.axis('off')

# 8. Imagem final com novo fundo
plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title("8. Composição Final")
plt.axis('off')

plt.tight_layout()
plt.show()
