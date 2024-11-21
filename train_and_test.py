import cv2
import numpy as np
from model import *
import os
import  pic_show
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time

EPOCH_NUM = 2
def pre_process_imp(input_path, save_dir, save_name, output_size):
    if not os.path.exists(input_path):
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(input_path)
    if str(save_dir).find('mask') != -1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     for r in img:
    #         for i in range(r.size):
    #             r[i] = 255 if r[i] > 0 else 0
    img = cv2.resize(img, output_size)
    # pic_show.show_pic(img)
    # if str(save_dir).find('mask') != -1:
    #     for r in img:
    #         for i in range(r.size):
    #             r[i] = 1 if r[i] > 127 else 0
    cv2.imwrite(os.path.join(save_dir, save_name), img)
    # pic_show.show_pic(img)
def pre_process():
    top = "./dataset"
    resized_top = "./resized_img"
    dirs = os.listdir(top)
    for dir in dirs:
        sub_dir = os.path.join(top,dir)
        files = os.listdir(sub_dir)
        for file in files:
            pre_process_imp(os.path.join(sub_dir, file), os.path.join(resized_top, dir), file, (1024, 768))

class MyDataSet(Dataset):
    def __init__(self, img_dir, mask_dir):
        super(MyDataSet, self).__init__()
        self.files_train_img = os.listdir(img_dir)
        self.files_train_mask = os.listdir(mask_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.files_train_img[item])
        mask_path = os.path.join(self.mask_dir, self.files_train_mask[item])
        image = cv2.imread(img_path)
        image = image.transpose((2,0,1))
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
        # print(image.shape," ",mask.shape)
        img = torch.tensor(image, dtype=torch.float, device=DEVICE)
        mask = torch.tensor(mask, dtype=torch.float, device=DEVICE)
        return img, mask

    def __len__(self):
        return len(self.files_train_mask)



if __name__ == "__main__":
    print("start time: ",time.ctime())
    # pre_process()
    train_set = MyDataSet("./resized_img/train_image", "./resized_img/train_mask")
    test_set = MyDataSet("./resized_img/val_image", "./resized_img/val_mask")
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn.to(device=DEVICE)
    # model = Unet()
    model = torch.load('unet.pt', DEVICE)
    # model.to(device=DEVICE)


    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    for epo in range(EPOCH_NUM):
        loss_epo = 0
        for img, mask in train_loader:
            output = model(img)
            output = output.squeeze(1)
            loss = loss_fn.forward(output, mask)
            loss_epo += loss.item()
            # if epo % 100 == 0:
            #     print(loss_epo)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"第{epo}轮训练，loss为 ： ",loss_epo)
    torch.save(model, "Unet.pt")
    model.eval()
    img, mask = next(iter(test_loader))
    output = model(img)
    img = img.squeeze(0)
    img_in_cpu = torch.Tensor.cpu(img).detach()
    img_np_array = np.array(img_in_cpu, dtype=np.uint8)
    img_np_array = img_np_array.transpose((1,2,0))
    output_in_cpu = torch.Tensor.cpu(output).detach()
    output_np_array = np.array(output_in_cpu)
    # print(output_np_array.shape)
    output_np_array = np.squeeze(output_np_array)
    if len(output_np_array.shape) > 2:
        for i in output_np_array.shape[0]:
            mask = output_np_array[i]
            cv2.imwrite(f"test_output_{i}.png", mask)
    else:
        cv2.imwrite("test_output.png", output_np_array)
    for r in output_np_array:
        for i in range(r.size):
            r[i] = 255 if r[i] > 0 else 0
    print("end time: ",time.ctime())
    # pic = np.hstack((img_np_array, output_np_array))
    pic_show.show_pic(img_np_array)
    pic_show.show_pic(output_np_array)