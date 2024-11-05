#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
def generate_ripple_with_fourier(size=512, frame=0, speed=0.05):
    # Criar uma grade de coordenadas
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Calcular a distância radial
    distance_from_center = np.sqrt(X**2 + Y**2)

    # Parâmetros dos ripples
    current_radius1 = frame * speed  # Raio do primeiro ripple
    current_radius2 = max(0, (frame - 10) * speed)  # Raio do segundo ripple começa 10 quadros depois

    # Criar a intensidade do primeiro ripple usando Fourier
    ripple1_freq = np.exp(-(current_radius1 - distance_from_center)**2 / 0.01)
    ripple1_freq = np.fft.fft2(ripple1_freq)  # Aplicar Transformada de Fourier

    # Criar a intensidade do segundo ripple
    ripple2_freq = 0.5 * np.exp(-(current_radius2 - distance_from_center)**2 / 0.02)
    ripple2_freq = np.fft.fft2(ripple2_freq)  # Aplicar Transformada de Fourier

    # Somar as duas frequências
    combined_freq = ripple1_freq + ripple2_freq

    # Aplicar a transformada inversa de Fourier
    ripple_final = np.fft.ifft2(combined_freq).real

    # Normalizar
    ripple_final = (ripple_final - ripple_final.min()) / (ripple_final.max() - ripple_final.min())

    # Aplicar uma máscara para que a intensidade diminua até a borda
    mask = np.clip(1 - distance_from_center, 0, 1)
    ripple_final = ripple_final * mask

    return ripple_final

#%%
#Teste da imagem
ripple = generate_ripple_with_fourier(512,10)
plt.imshow(ripple, cmap='gray', vmin=0, vmax=1)

#%%
# Geração das imagens
num_frames = 90
size = 512
output_folder = 'ripple_frames_fourier'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for frame in range(num_frames):
    ripple = generate_ripple_with_fourier(size=size, frame=frame)
    plt.imshow(ripple, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(f'{output_folder}/ripple_frame_{frame:03d}.png', bbox_inches='tight', pad_inches=0)
