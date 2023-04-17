#!/usr/bin/env python3

import numpy as np
import time 
import logging
import os
import random
import torch
import torch.utils.data

import pandas as pd 
import csv

from renderer.online_object_renderer import OnlineObjectRenderer
from utils import utils


def visualization(color, depth, pc):

    # visualization for your debugging
    import matplotlib.pyplot as plt
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(color[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show depth image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(depth)
    ax.set_title('depth image')
        
    # up to now, suppose you get the points box as pbox. Its shape should be (5280, 3)
    # then you can use the following code to visualize the points in pbox
    # You shall see the figure in the homework assignment
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='.', color='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D ploud cloud')
                  
    plt.show()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        subsample,
        gt_filename,
        #pc_size=1024,
    ):

        self.data_source = data_source 
        self.subsample = subsample
        self.split_file = split_file
        self.gt_filename = gt_filename
        #self.pc_size = pc_size
        
        # initialize the renderer
        self.renderer = OnlineObjectRenderer(caching=True)
        self.all_poses = utils.uniform_quaternions()        

        # example
        # data_source: "data"
        # ws.sdf_samples_subdir: "SdfSamples"
        # self.gt_files[0]: "acronym/couch/meshname/sdf_data.csv"
            # with gt_filename="sdf_data.csv"

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):     
        return NotImplementedError

    def sample_pointcloud(self, csvfile, pc_size):
        # f=pd.read_csv(csvfile, sep=',',header=None).values
        # f = f[f[:,-1]==0][:,:3]
        
        # use partial view
        cad_path = csvfile.replace('sdf_data.csv', 'model.obj')
        cad_scale = 1.0
        
        # read centroid
        filename = csvfile.replace('sdf_data', 'centroid')
        centroid = pd.read_csv(filename, sep=',',header=None).values
        
        # change object
        self.renderer.change_object(cad_path, cad_scale, centroid)
    
        fail = 0
        while 1:
            viewing_index = np.random.randint(0, high=len(self.all_poses))
            camera_pose = self.all_poses[viewing_index]
    
            color, depth, pc, transferred_pose = self.renderer.render(camera_pose)
            pc = pc.dot(utils.inverse_transform(transferred_pose).T)[:, :3]
            
            if pc.shape[0] > 200:
                break
            else:
                fail += 1
                print('fail', fail)
                
                if fail > 10:
                    print(csvfile)
                    visualization(color, depth, pc)
        # visualization(color, depth, pc)

        f = pc.copy()
        if f.shape[0] < pc_size:
            pc_idx = np.random.choice(f.shape[0], pc_size)
        else:
            pc_idx = np.random.choice(f.shape[0], pc_size, replace=False)

        return torch.from_numpy(f[pc_idx]).float()


    def labeled_sampling(self, csvfile, subsample, pc_size=1024):
        f = pd.read_csv(csvfile, sep=',', header=None).values
        f = torch.from_numpy(f)

        half = int(subsample / 2) 
        neg_tensor = f[f[:,-1]<0]
        pos_tensor = f[f[:,-1]>0]

        if pos_tensor.shape[0] < half:
            pos_idx = torch.randint(pos_tensor.shape[0], (half,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half]

        if neg_tensor.shape[0] < half:
            neg_idx = torch.randint(neg_tensor.shape[0], (half,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half]

        pos_sample = pos_tensor[pos_idx]
        neg_sample = neg_tensor[neg_idx]

        # sample point cloud
        # pc = f[f[:,-1]==0][:,:3]
        # pc_idx = torch.randperm(pc.shape[0])[:pc_size]
        # pc = pc[pc_idx]
        pc = self.sample_pointcloud(csvfile, pc_size)

        samples = torch.cat([pos_sample, neg_sample], 0)

        return pc.float().squeeze(), samples[:,:3].float().squeeze(), samples[:, 3].float().squeeze() # pc, xyz, sdv


    def get_instance_filenames(self, data_source, split, gt_filename="sdf_data.csv"):
            csvfiles = []
            for dataset in split: # e.g. "acronym" "shapenet"
                for class_name in split[dataset]:
                    for instance_name in split[dataset][class_name]:
                        instance_filename = os.path.join(data_source, dataset, class_name, instance_name, gt_filename)
                        
                        if not os.path.isfile(instance_filename):
                            logging.warning("Requested non-existent file '{}'".format(instance_filename))
                            continue

                        csvfiles.append(instance_filename)
            return csvfiles
