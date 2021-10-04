import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio
from sklearn.model_selection import train_test_split
from data import read_image, noisy_image
from  model_design import DenoiseAutoEncoder
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data_path = r"E:\迅雷下载\stl10_binary\train_X.bin"
    images = read_image(data_path)
    print(f"image.shape: {images.shape}")

    images_noisy = noisy_image(images, 30)
    print(f"image_noise: {images_noisy.min()} ~ {images_noisy.max()}")

    data_Y = np.transpose(images, axes=(0, 3, 2, 1))
    data_X = np.transpose(images_noisy, axes=(0, 3, 2, 1))

    X_train, X_valid, y_train, y_valid = train_test_split(data_X, data_Y, test_size=0.2, random_state=123)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.float32)
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_valid.shape: {X_valid.shape}, y_valid.shape: {y_valid.shape}")

    image_index = 1
    img = X_valid[image_index]
    img = img.unsqueeze(0)
    img_noisy = np.transpose(img.data.numpy(), axes=(0, 3, 2, 1))
    img_noisy = img_noisy[0, ...]

    model = DenoiseAutoEncoder()
    checkpoint = torch.load("./checkpoint/denois_auto_encoder.pth")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _, output = model(img)
    img_denoise = np.transpose(output.data.numpy(), (0, 3, 2, 1))
    img_denoise = img_denoise[0, ...]

    img_ori = y_valid[image_index]
    img_ori = img_ori.unsqueeze(0)
    img_ori = np.transpose(img_ori.data.numpy(), (0, 3, 2, 1))
    img_ori = img_ori[0, ...]
    print(f"Noisy image PSNR: {peak_signal_noise_ratio(img_ori, img_noisy)} dB")
    print(f"Denoise image PSNR: {peak_signal_noise_ratio(img_ori, img_denoise)} dB")

    show_list = ["img_ori", "img_noisy", "img_denoise"]
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(eval(show_list[i]))
        plt.title(show_list[i])
    plt.tight_layout()
    plt.show()

