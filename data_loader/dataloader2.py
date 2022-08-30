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
    def __init__(self, data_path: str = None, transforms=None, mode: str = "train", image_size: int = 256,
                 k_shots: int = 1):
        self.k_shots = k_shots
        self.image_size = image_size
        self.data_path = data_path
        self.mode = mode

        self.imgs_path = glob.glob(data_path + "/imgs/*.jpg")
        self.masks_path = glob.glob(data_path + "/masks/*.jpg")
        image_name = [os.path.basename(path) for path in self.imgs_path]
        mask_name = [os.path.basename(path) for path in self.masks_path]
        self.inter_name = list(set(image_name) & set(mask_name))

        self.imgdir = os.path.dirname(self.imgs_path[0])
        self.maskdir = os.path.dirname(self.masks_path[0])

        if transforms is None:
            self.transforms = A.Compose([
                A.Resize(image_size, image_size, always_apply=True),
                ToTensorV2()
            ])
        else:
            self.transforms = transforms
        self.split_data()
        self.test_video = self._divide_video()

    def __len__(self) -> int:
        if self.mode == 'test':
            return len(self.inter_name_test)
        elif self.mode == 'val':
            return len(self.inter_name_val)
        elif self.mode == 'train':
            return len(self.data_2C)
        elif self.mode == 'video':
            return len(self.test_video)

    def __getitem__(self, idx):
        if self.mode == 'test':
            name = self.inter_name_test[idx]
            clasS = 2 if name.find('3C__') != -1 else 3
            
            support_names = []

            while True:  # keep sampling support set if query == support
                support_name = random.choice(self.inter_name_test)
                support_class = 2 if support_name.find('3C__') != -1 else 3
                if name != support_name and support_class==clasS: support_names.append(support_name)
                if len(support_names) == self.k_shots: break
        
        elif self.mode == 'val':
            name = self.inter_name_val[idx]
            clasS = 2 if name.find('3C__') != -1 else 3
            
            support_names = []

            while True:  # keep sampling support set if query == support
                support_name = random.choice(self.inter_name_val)
                support_class = 2 if support_name.find('3C__') != -1 else 3
                if name != support_name and support_class==clasS: support_names.append(support_name)
                if len(support_names) == self.k_shots: break

        elif self.mode =='train':
            name = self.data_2C[idx]
            clasS = 1
            support_names = []

            while True:  # keep sampling support set if query == support
                support_name = random.choice(self.data_2C)
                if name != support_name: support_names.append(support_name)
                if len(support_names) == self.k_shots: break
        
        elif self.mode == 'video':
            name = self.test_video[idx]
            clasS = 0
            support_names = []
            while True:  # keep sampling support set if query == support
                support_name = random.choice(self.test_video)
                if name != support_name: support_names.append(support_name)
                if len(support_names) == self.k_shots: break

            


        

        image = cv2.imread(os.path.join(self.imgdir, name))
        mask = cv2.imread(os.path.join(self.maskdir, name), 0)
        transformed = self.transforms(image=image, mask=mask)
        image_transformed, mask_transformed = transformed["image"] / 255, transformed["mask"] / 255

        support_images = []
        support_masks = []
        for i in range(self.k_shots):
            support_image = cv2.imread(os.path.join(self.imgdir, support_names[i]))
            support_mask = cv2.imread(os.path.join(self.maskdir, support_names[i]), 0)
            transformed_supp = self.transforms(image=support_image, mask=support_mask)
            image_transformed_supp, mask_transformed_supp = transformed_supp["image"] / 255, transformed_supp["mask"] / 255
            support_images.append(image_transformed_supp.float())
            support_masks.append(mask_transformed_supp.float())
        
        return {
            'support_images': support_images, 
            'support_mask':support_masks, 
            'query_images':image_transformed.float(), 
            'query_labels':mask_transformed.float(),
            'class_ids': clasS,
            'name': name
        }

    def split_data(self):
        self.data_4C = []
        self.data_3C = []
        self.data_2C = []
        for idx, data in enumerate(self.inter_name):
            if data.find('2C__') != -1: self.data_2C.append(data)
            elif data.find('3C__') != -1: self.data_3C.append(data)
            elif data.find('4C__') != -1: self.data_4C.append(data)
        
        # self.val_data = []
        # for idx, data in enumerate(self.data_2C):
        #     if idx < 500: self.val_data.append(data)
        
        # for idx, data in enumerate(self.val_data):
        #     self.data_2C.remove(data)
        
        self.data_3C.sort()
        self.data_4C.sort()
        del self.inter_name

        self.inter_name_test = []
        self.inter_name_val = []
        self.data_2C_test = []
        for idx, data in enumerate(self.data_3C):
            if idx <=500: 
                self.inter_name_val.append(data)
                # self.data_2C_test.append(data)
            else: 
            #     self.inter_name_test.append(data)
                self.data_2C_test.append(data)
        
        for idx, data in enumerate(self.data_4C):
            if idx <=500: 
                self.inter_name_val.append(data)
                # self.data_2C_test.append(data)
            # else: 
            # #     self.inter_name_test.append(data)
            #     self.data_2C_test.append(data)


    
    def _divide_video(self):
        video = defaultdict(list)
        for file_name in self.data_2C_test:
            # split file name by last occurrence '_'
            video_name = file_name.rsplit('_', 1)[0]
            video[video_name].append(file_name)
        # split video into train, val, test by self.ratio
        video_list = list(video.keys())
        random.shuffle(video_list)
        # merge list of video into one list
        videos = [video[video_name] for video_name in video_list]
        video_items = [item for sublist in videos for item in sublist]
        print('video: ', len(video_list))


        return video_items
    

        
    
class UETDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 2, num_workers: int = 1, image_size=256, k_shots:int=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.image_size = image_size
        self.k_shots = k_shots

    def setup(self, stage: Optional[str] = None):
        if not os.path.exists(self.data_path):
            download_and_extract_from_google_drive(UET_ECHO, self.data_path)
        datasets = {
            x: UETDataset(data_path=self.data_path, mode=x, image_size=self.image_size, k_shots=self.k_shots) for x in ["train", "test", "val", "video"]
        }
        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]
        self.video = datasets["video"]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False, shuffle=False)
    
    def test_video(self) -> DataLoader:
        return DataLoader(self.video, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=False, shuffle=False)
