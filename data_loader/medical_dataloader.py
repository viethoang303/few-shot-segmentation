import glob
import os
import random
from collections import defaultdict
from typing import Optional

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from data_registry import UET_ECHO
from utils import download_and_extract_from_google_drive


class UETDataset(Dataset):
    def __init__(self, data_path: str = None, transforms=None, mode: str = "train", image_size: int = 256,
                 ratio: dict = {"train": 0.8, "val": 0.1, "test": 0.1}):
        self.data_path = data_path
        self.mode = mode
        self.ratio = ratio
        self.imgs_path = glob.glob(data_path + "/imgs/*.jpg")
        self.masks_path = glob.glob(data_path + "/masks/*.jpg")
        image_name = [os.path.basename(path) for path in self.imgs_path]
        mask_name = [os.path.basename(path) for path in self.masks_path]
        self.inter_name = list(set(image_name) & set(mask_name))
        self.inter_name = self._divided_data_by_video()
        self.imgdir = os.path.dirname(self.imgs_path[0])
        self.maskdir = os.path.dirname(self.masks_path[0])

        if transforms is None:
            self.transforms = A.Compose([
                A.Resize(image_size, image_size, always_apply=True),
                ToTensorV2()
            ])
        else:
            self.transforms = transforms

    def _divided_data_by_video(self):

        video = defaultdict(list)
        for file_name in self.inter_name:
            # split file name by last occurrence '_'
            video_name = file_name.rsplit('_', 1)[0]
            video[video_name].append(file_name)
        # split video into train, val, test by self.ratio
        video_list = list(video.keys())
        random.shuffle(video_list)
        train_video = video_list[:int(len(video_list) * self.ratio["train"])]
        val_video = video_list[int(len(video_list) * self.ratio["train"]):int(
            len(video_list) * (self.ratio["train"] + self.ratio["val"]))]
        test_video = video_list[int(len(video_list) * (self.ratio["train"] + self.ratio["val"])):]
        # merge list of video into one list
        train_video = [video[video_name] for video_name in train_video]
        val_video = [video[video_name] for video_name in val_video]
        test_video = [video[video_name] for video_name in test_video]
        train_video = [item for sublist in train_video for item in sublist]
        val_video = [item for sublist in val_video for item in sublist]
        test_video = [item for sublist in test_video for item in sublist]
        print(train_video)
        return {
            "train": train_video,
            "val": val_video,
            "test": test_video
        }

    def __len__(self) -> int:
        return len(self.inter_name[self.mode])

    def __getitem__(self, idx) -> (torch.tensor, torch.tensor):
        name = self.inter_name[self.mode][idx]
        image = cv2.imread(os.path.join(self.imgdir, name), 0)
        mask = cv2.imread(os.path.join(self.maskdir, name), 0)

        transformed = self.transforms(image=image, mask=mask)
        image_transformed, mask_transformed = transformed["image"] / 255, transformed["mask"] / 255
        mask_transformed = mask_transformed[None, ...]
        foreground = torch.where(mask_transformed > 0.5, torch.tensor(1), torch.tensor(0))
        background = torch.where(mask_transformed <= 0.5, torch.tensor(1), torch.tensor(0))
        mask_transformed = torch.cat((background, foreground), dim=0)
        return image_transformed.float(), mask_transformed.float()


class UETDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 2, num_workers: int = 1, image_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        if not os.path.exists(self.data_path):
            download_and_extract_from_google_drive(UET_ECHO, self.data_path)
        datasets = {
            x: UETDataset(data_path=self.data_path, mode=x, image_size=self.image_size) for x in ["train", "val", "test"]
        }
        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


if __name__ == "__main__":
    datapath = "data/uet_echo"
    batch_size = 2
    num_workers = 1
    DataModule = UETDataModule("data/uet_echo", batch_size=batch_size, num_workers=num_workers)
    DataModule.setup()
    train_loader = DataModule.train_dataloader()
    from matplotlib import pyplot as plt

    for i, (image, mask) in enumerate(train_loader):
        print("image shape", image.shape)
        print("mask shape", mask.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(image[0, 0, ...].numpy(), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(mask[0, 1, ...].numpy(), cmap="gray")
        plt.show()
        if i == 2:
            break
    print(f"[INFO] DataModule: {DataModule.__dict__}")
