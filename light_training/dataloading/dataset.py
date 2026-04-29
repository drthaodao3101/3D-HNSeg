# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Modefined by Selab 

from sklearn.model_selection import KFold  ## K折交叉验证
import pickle
import os
import json
import math
import numpy as np
import torch
from monai import transforms
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
import glob 
from light_training.dataloading.utils import unpack_dataset
import random 
import blosc2
from pathlib import Path

class MedicalDataset(Dataset):
    def __init__(self,data_dir, datalist, test=False) -> None:
        super().__init__()
        self.data_dir=data_dir
        self.datalist = datalist
        self.test = test 

        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(f"{p}.pkl")
            self.data_cached.append(info)

        print(f"data length is {len(self.datalist)}")
        
    def load_pkl(self, properties_path):
        pass 
        #properties_path = f"{data_path[:-4]}.pkl"
        df = open(properties_path, "rb")
        info = pickle.load(df)
        return info 
    
    def post(self, batch_data):
        return batch_data
    
    def read_data(self, data_path):
        
        volume_fp = data_path+".b2nd"
        label_fp = data_path+"_seg.b2nd"

        image_data = blosc2.open(volume_fp, mode='r')[:]                    #   C*D*H*W
        seg_data = None 
        if not self.test:
            seg_data = blosc2.open(label_fp, mode='r')[:]
        return image_data, seg_data

    def __getitem__(self, i):
        
        image, seg = self.read_data(self.datalist[i])
        properties = self.data_cached[i]

        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)


class MedicalDataset_ForValidation(Dataset):
    def __init__(self,data_dir, datalist, test=False) -> None:
        super().__init__()
        self.data_dir=data_dir
        self.datalist = datalist
        self.test = test 

        self.data_cached = []
        for p in tqdm(self.datalist, total=len(self.datalist)):
            info = self.load_pkl(f"{p}.pkl")
            self.data_cached.append(info)

        print(f"data length is {len(self.datalist)}")
        
    def load_pkl(self, properties_path):
        pass 
        df = open(properties_path, "rb")
        info = pickle.load(df)
        return info 
    
    def post(self, batch_data):
        return batch_data
    
    def read_data(self, data_path):
        
        volume_fp = data_path+".b2nd"

        label_fp= "./data/henect204_seg/gt_segmentations/" +Path(data_path).name+".nrrd"
        #label_fp = data_path+"_seg.b2nd"

        image_data = blosc2.open(volume_fp, mode='r')[:]                    #   C*D*H*W
        seg_data = None 
        if not self.test:
            #seg_data = blosc2.open(label_fp, mode='r')[:]
            img = sitk.ReadImage(label_fp)
            seg_data = sitk.GetArrayFromImage(img)
        return image_data, seg_data

    def __getitem__(self, i):
        
        image, seg = self.read_data(self.datalist[i])

        properties = self.data_cached[i]
        properties['name']=os.path.basename(self.datalist[i])
        if seg is None:
            return {
                "data": image,
                "properties": properties
            }
        else :
            return {
                "data": image,
                "seg": seg,
                "properties": properties
            }

    def __len__(self):
        return len(self.datalist)



def get_train_val_loader_from_json(data_root,json_path,fold):                  #For train

    data_dir=os.path.join(data_root,"Plans_3d_lowres")
    assert os.path.exists(json_path), f"JSON file not found: {json_path}"
    with open(json_path) as f:
        js_data = json.load(f)
        train_datalist, val_datalist = js_data[fold]["train"], js_data[fold]["val"]

    def add_pre(datalist):
        for i in range(len(datalist)):
            datalist[i] = os.path.join(data_dir, datalist[i])

    add_pre(train_datalist)
    add_pre(val_datalist)

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")

    train_ds = MedicalDataset(data_dir,train_datalist)
    val_ds = MedicalDataset(data_dir,val_datalist)

    loader = [train_ds, val_ds]
    return loader

def get_train_val_loader_from_json_for_validation(data_root,json_path,fold):    #For validation                  

    data_dir=os.path.join(data_root,"Plans_3d_lowres")
    assert os.path.exists(json_path), f"JSON file not found: {json_path}"
    with open(json_path) as f:
        js_data = json.load(f)
        _, val_datalist = js_data[fold]["train"], js_data[fold]["val"]

    def add_pre(datalist):
        for i in range(len(datalist)):
            datalist[i] = os.path.join(data_dir, datalist[i])

    add_pre(val_datalist)

    print(f"validation data is {len(val_datalist)}")

    val_ds = MedicalDataset_ForValidation(data_dir,val_datalist)

    return val_ds

def get_train_val_loader_from_json_for_cross_validation(data_root,json_path):   #For computing metrics                  

    data_dir=os.path.join(data_root,"Plans_3d_lowres")
    assert os.path.exists(json_path), f"JSON file not found: {json_path}"
    with open(json_path) as f:
        js_data = json.load(f)
        val_datalist = js_data[0]["val"]+js_data[1]["val"]+js_data[2]["val"]+js_data[3]["val"]+js_data[4]["val"]

    def add_pre(datalist):
        for i in range(len(datalist)):
            datalist[i] = os.path.join(data_dir, datalist[i])

    add_pre(val_datalist)

    print(f"validation data is {len(val_datalist)}")

    val_ds = MedicalDataset_ForValidation(data_dir,val_datalist)

    return val_ds

