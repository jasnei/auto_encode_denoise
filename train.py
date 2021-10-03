import numpy as np
import torch
from sklearn.model_selection import train_test_split
import hiddenlayer as hl
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from data import read_image, noisy_image
from  model_design import DenoiseAutoEncoder


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

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = DenoiseAutoEncoder()
    model.cuda()
    LR = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    history_1 = hl.History()
    canvas_1 = hl.Canvas()

    train_num = 0
    valid_num = 0

    print("training...")
    for epoch in range(10):
        print(f">>>epoch: {epoch + 1}")
        train_loss_epoch = 0
        valid_loss_epoch = 0

        model.train()
        for step, (datas, labels) in enumerate(train_loader):
            datas = datas.cuda()
            labels = labels.cuda()
            _, output = model(datas)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            train_num = train_num + datas.size(0)

        model.eval()
        for step, (datas, labels) in enumerate(valid_loader):
            datas = datas.cuda()
            labels = labels.cuda()
            _, output = model(datas)
            loss = loss_fn(output, labels)
            valid_loss_epoch += loss.item()
            valid_num = valid_num + datas.size(0)

        train_loss = train_loss_epoch / train_num
        valid_loss = valid_loss_epoch / valid_num
        history_1.log(epoch, train_loss=train_loss, valid_loss=valid_loss)
        with canvas_1:
            canvas_1.draw_plot([history_1["train_loss"], history_1["valid_loss"]])

    torch.save({"state_dict": model.state_dict(),}, "./checkpoint/denois_auto_encoder.pth")
    print('Save model done!')

    

