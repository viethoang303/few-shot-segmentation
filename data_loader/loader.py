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
from data_loader.data_registry import UET_ECHO
from data_loader.utils import download_and_extract_from_google_drive


class UETDataset(Dataset):
    def __init__(self, data_path: str = None, transforms=None, image_size: int = 256,
                 ratio: dict = {"train": 0.8, "val": 0.1, "test": 0.1}):
        self.data_path = data_path
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
        # merge list of video into one list
        videos = [video[video_name] for video_name in video_list]
        video_items = [item for sublist in videos for item in sublist]

        return video_items

    def __len__(self) -> int:
        return len(self.inter_name)

    def __getitem__(self, idx):
        name = self.inter_name[idx]
        image = cv2.imread(os.path.join(self.imgdir, name))
        mask = cv2.imread(os.path.join(self.maskdir, name), 0)

        transformed = self.transforms(image=image, mask=mask)
        image_transformed, mask_transformed = transformed["image"] / 255, transformed["mask"] / 255
        # mask_transformed = mask_transformed[None, ...]
        # foreground = torch.where(mask_transformed > 0.5, torch.tensor(1), torch.tensor(0))
        # background = torch.where(mask_transformed <= 0.5, torch.tensor(1), torch.tensor(0))
        # mask_transformed = torch.cat((background, foreground), dim=0)
        return {
            'name':name, 
            'image':image_transformed.float(), 
            'mask': mask_transformed.float()
        }


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
        self.dataset = UETDataset(data_path=self.data_path, image_size=self.image_size)
    def dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

