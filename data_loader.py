import os
import glob
import numpy as np
import torch
import pathlib
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
)
from monai.data import Dataset
import h5py
import threading
import torchvision.transforms as transforms


class SliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """


    def __init__(self, root,transforms = None, mode='train', train_test_split = 0.8):

        files = list(pathlib.Path(root).iterdir())
        files = sorted([str(i) for i in files])
        imgs=[]
        self.xfms = transforms
        total_pats= len(files)
        print("TOTAL PATS",total_pats,"and",total_pats*10,"files")
        # if mode == "train":
        #     for filename in files[:int(train_test_split * len(files))]:

        #         if filename[-3:] == '.h5':

        #             imgs.append(filename)

        # elif mode == 'test' or 'validation':
        #     for filename in files[int(train_test_split * len(files)):]:
        #         if filename[-3:] == '.h5':
        #             imgs.append(filename)

        if mode == "train":
            print("in trian")
            for filename in files[:int(train_test_split * total_pats)]:
                imgs.append(filename)
        elif mode == "train_inf":
            for filename in files[:int(train_test_split * total_pats)]:
                imgs.append(filename)
        elif mode =='val':
            print("into val")
            # print(files[int(train_test_split * total_pats):int(train_test_split * total_pats) + int(0.1 * total_pats)])
            
            for filename in files[int(train_test_split * total_pats):int(train_test_split * total_pats) + int(0.1 * total_pats)]:
                # print("val filename",filename)
                imgs.append(filename)
        # print("before")
        
        elif mode == 'test':
            print("in test")
            print(files[int(train_test_split * total_pats) + int(0.1 * total_pats):])
            #print(len(files[int(train_test_split * total_pats) + int(0.1 * total_pats):]))   
            for filename in files[int(train_test_split * total_pats) + int(0.1 * total_pats):]:
                # if filename not in ['ProstateX-0309.h5','ProstateX-0311.h5','ProstateX-0312.h5','ProstateX-0320.h5','ProstateX-0323.h5','ProstateX-0330.h5','ProstateX-0332.h5','ProstateX-0335.h5','ProstateX-0338.h5','ProstateX-0342.h5']:
                print("filename",filename)
                imgs.append(filename)
        else:
            print("INVALID MODE")

        self.examples = []


        for fname in imgs:
            with h5py.File(fname,'r') as hf:
                fsvol = hf['T2']
                num_slices = fsvol.shape[-1]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
        final_dic ={}
        
        with h5py.File(fname, 'r') as data:
            t2 = torch.from_numpy(data['T2'][0,:,:,:].astype(np.float64))
            adc = torch.from_numpy(data['ADC'][0,:,:,:].astype(np.float64))
            # dwi0 = torch.from_numpy(data['DWI0'][0,:,:,:].astype(np.float64))
            pd = torch.from_numpy(data['PD'][0,:,:,:].astype(np.float64))
            dce_01 = torch.from_numpy(data['DCE_01'][0,:,:,:].astype(np.float64))
            dce_02 = torch.from_numpy(data['DCE_02'][0,:,:,:].astype(np.float64))
            dce_03 = torch.from_numpy(data['DCE_03'][0,:,:,:].astype(np.float64))
            
        dict_ = {"T2": t2,"ADC":adc,
                 "PD": pd, 
                 "DCE_01":dce_01,
                "DCE_02": dce_02, "DCE_03":dce_03}
        
        trans_dic = dict_ # self.xmfs(dict_)

        t2 = trans_dic['T2'][None,:,:,slice]
        adc = trans_dic['ADC'][None,:,:,slice]
        # dwi0 = trans_dic['DWI0'][None,:,:,slice]
        pd = trans_dic['PD'][None,:,:,slice]
        dce_01 = trans_dic['DCE_01'][None,:,:,slice]
        dce_02 = trans_dic['DCE_02'][None,:,:,slice]
        dce_03 = trans_dic['DCE_03'][None,:,:,slice]
        
        dce_02_crop = transforms.CenterCrop(60)(dce_02)
        dce_03_crop = transforms.CenterCrop(60)(dce_03)
        return t2,pd,adc,dce_01,dce_02,dce_03,str(fname.split('/')[-1]), slice
#         data__ = {"T2": torch.from_numpy(t2),"ADC":torch.from_numpy(adc),
#                 "PD": torch.from_numpy(pd), "DCE_01":torch.from_numpy(dce_01),
#                 "DCE_02":torch.from_numpy(dce_02), "DCE_03":torch.from_numpy(dce_03)}
        
        # data_lst = {'A':torch.concatenate((t2,adc,pd,dce_01),axis=0),'B':torch.concatenate((dce_02, dce_03), axis =0), 'DX': i } #, 'C':torch.concatenate((dce_02_crop, dce_03_crop),axis=0)}
        
        
        # return data_lst