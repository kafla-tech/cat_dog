from operator import index

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm
def main():
    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    class Datasets(torch.utils.data.Dataset):
        def __init__(self, path_dir1:str, path_dir2:str):
            super().__init__()
            self.path_dir1 = path_dir1
            self.path_dir2 = path_dir2
            self.dir1_list = sorted([f for f in os.listdir(path_dir1)
                                    if os.path.splitext(f)[1].lower() in VALID_EXTS])
            self.dir2_list = sorted([f for f in os.listdir(path_dir2)
                                    if os.path.splitext(f)[1].lower() in VALID_EXTS])
            
        
        def __len__(self):
            return len(self.dir1_list) + len(self.dir2_list)


        def __getitem__(self, index):
            if index < len(self.dir1_list):
                class_id = 0
                img_path = os.path.join(self.path_dir1, self.dir1_list[index])
            else:
                class_id = 1
                new_index = index - len(self.dir1_list)
                img_path = os.path.join(self.path_dir2, self.dir2_list[new_index])

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) 
            img = img / 255.0
            img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
            img = img.transpose((2, 0, 1))
            
            t_img = torch.from_numpy(img)
            
            t_class_id = torch.tensor(class_id)
            return {'img': t_img, 'class_id': t_class_id}
        

    train_dogs = '/home/kafla/Projects/cat or dogs/cat-and-dog/versions/1/training_set/training_set/dogs/'
    train_cats = '/home/kafla/Projects/cat or dogs/cat-and-dog/versions/1/training_set/training_set/cats/'
    test_dogs = '/home/kafla/Projects/cat or dogs/cat-and-dog/versions/1/test_set/test_set/dogs/'
    test_cats = '/home/kafla/Projects/cat or dogs/cat-and-dog/versions/1/test_set/test_set/cats/'


    train_ds_catdog = Datasets(train_dogs, train_cats)
    test_ds_catdog = Datasets(test_dogs, test_cats)

    len(train_ds_catdog)
    len(test_ds_catdog)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        train_ds_catdog,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds_catdog,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    class CNV(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.activation = nn.LeakyReLU(0.2)
            self.maxpool = nn.MaxPool2d(2, stride=2)
            self.conv0 = nn.Conv2d(3, 32, 3, stride=1, padding=0)
            self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
            self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
            
            self.adaptiev = nn.AdaptiveAvgPool2d((1, 1))
            self.flaten = nn.Flatten()
            self.linear1 = nn.Linear(64, 10)
            self.linear2 = nn.Linear(10, 2)
            
        def forward(self, x):
            
            out = self.conv0(x)
            out = self.activation(out)
            out = self.maxpool(out)
            
            out = self.conv1(out)
            out = self.activation(out)
            out = self.maxpool(out)
            
            out = self.conv2(out)
            out = self.activation(out)
            out = self.maxpool(out)

            out = self.conv3(out)
            out = self.activation(out)
            out = self.maxpool(out)
            
            out = self.conv4(out)
            out = self.activation(out)
            out = self.maxpool(out)

            out = self.adaptiev(out)
            out = self.flaten(out)
            out = self.linear1(out)
            out = self.activation(out)
            out = self.linear2(out)
            
            return out

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    model = CNV()



    for sample in train_loader:
        img = sample['img']
        label = sample['class_id']
        model(img)
        break


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    checkpoint_path = "checkpoint.pth"

    start_epoch = 0

    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state'])

        optimizer.load_state_dict(checkpoint['optimizer_state'])

        start_epoch = checkpoint['epoch'] + 1

        print(f"Продолжаем с эпохи {start_epoch}")

    else:
        print("Новая модель")
        
    def accuracy(pred, label):

        answer = pred.detach().argmax(1) == label

        return answer.float().mean()


    epochs = 20

    for epoch in range(start_epoch, epochs):

        model.train()

        loss_val = 0
        acc_val = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}"
        )

        for sample in pbar:

            img = sample['img']
            label = sample['class_id']

            optimizer.zero_grad()

            pred = model(img)

            loss = loss_fn(pred, label)

            loss.backward()

            optimizer.step()

            loss_val += loss.item()

            acc_val += accuracy(pred, label).item()

        print(
            f"Epoch {epoch +1}: "
            f"Loss = {loss_val/len(train_loader):.4f}, "
            f"Accuracy = {acc_val/len(train_loader):.4f}"
        )

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss_val / len(train_loader),
            'accuracy': acc_val / len(train_loader)
        }, checkpoint_path)

        print("Checkpoint сохранён")

    object1 = input("введите данные 1 обьекта: ")
    object2 = input("введите данные 2 обьекта:")
    
    classes = [object1, object2]

    def predict_folder(folder_path):

        model.eval()

        files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]

        if len(files) == 0:
            print("В папке нет изображений")
            return

        for file in files:

            path = os.path.join(folder_path, file)

            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Ошибка чтения: {path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32) / 255.0

            img = cv2.resize(img, (100, 100))

            img = img.transpose((2, 0, 1))

            img_tensor = torch.from_numpy(img).unsqueeze(0)

            with torch.no_grad():

                pred = model(img_tensor)

                probs = torch.softmax(pred, dim=1)[0]

            print(f"\nФайл: {file}")

            for i, prob in enumerate(probs):

                print(f"{classes[i]}: {prob.item() * 100:.2f}%")

            index = torch.argmax(probs).item()

            print(f"Это: {classes[index]}")

    predict_folder('/home/kafla/Projects/cat or dogs/test')
            

if __name__ == "__main__":
    main()