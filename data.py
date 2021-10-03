import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise


def read_image(data_path):
    with open(data_path, 'rb') as f:
        data_1 = np.fromfile(f, dtype=np.uint8)

        images = np.reshape(data_1, (-1, 3, 96, 96))
        images = np.transpose(images, axes=(0, 3, 2, 1))

    return images / 255.0


def noisy_image(images, sigma=1):
    sigma_2 = sigma ** 2 / (255**2)
    images_noisy = np.zeros_like(images)
    for i in range(images.shape[0]):
        image = images[i]
        noise_im = random_noise(image, mode='gaussian', var=sigma_2, clip=True)
        images_noisy[i] = noise_im

    return images_noisy


if __name__ == '__main__':

    data_path = r"E:\迅雷下载\stl10_binary\train_X.bin"
    images = read_image(data_path)
    print(f"image.shape: {images.shape}")

    images_noisy = noisy_image(images, 30)
    print(f"image_noise: {images_noisy.min()} ~ {images_noisy.max()}")

    plt.figure(figsize=(6, 6))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.imshow(images[i, ...])
        plt.axis('off')
    plt.title("Original")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.imshow(images_noisy[i, ...])
        plt.axis('off')
    plt.title("Noisy Images")
    plt.tight_layout()
    plt.show()
