# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import os, time, sys
import os.path
import numpy as np
from transforms3d.euler import quat2euler
from transforms3d.quaternions import *
from imgaug import augmenters as iaa
import cv2
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from ycb_render.ycb_renderer import *
import torch.nn.functional as F
from pathlib import Path
import json
import imageio

from cosypose.lib3d import Transform


def load_depth(path):
  """Loads a depth image from a file.

  :param path: Path to the depth image file to load.
  :return: ndarray with the loaded depth image.
  """
  d = imageio.imread(path)
  return d.astype(np.float32)

class synpick_dataset(data.Dataset):
    def __init__(self, class_ids, object_names, class_model_num, dataset_path, 
                sequence_id, prediction_id, cosypose_results_path):
        '''
        class_ids :   class_ids=[0] This is always the same
        object_names = [002_master_chef_can]
        class_model_num: 1 This is always the same
        path: ../YCB_Video_Dataset/data/ This is always the same
        list_file: test_list_file = './datasets/YCB/{}/seq{}.txt'.format(target_obj, args.n_seq)
        '''

        # loads all the frames in a sequece.
        self.dataset_path = dataset_path
        self.sequence_id = sequence_id
        self.prediction_id = prediction_id

        self.sequence_path = Path(self.dataset_path) / self.sequence_id

        assert self.sequence_path.exists(), f'Sequence {self.sequence_id} does not exists in {self.dataset_path}'
        count = 0
        for img in (self.sequence_path / 'rgb').iterdir():
            count += 1
        self.num_files = count 

        print('***CURRENT SEQUENCE INCLUDES {} IMAGES***'.format(self.num_files))

        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.class_names_all = class_name_file.read().split('\n')

        assert len(object_names)==1, "current only support loading the information for one object !!!"
        self.object_names = object_names
        self.class_ids = class_ids
        self.class_model_number = class_model_num

        # load scene_gt.json and scene_gt_info.json
        with open(self.sequence_path / 'scene_gt.json') as gt_file:
            self.scene_gt = json.load(gt_file)
        with open(self.sequence_path / 'scene_gt_info.json') as gt_info_file:
            self.scene_gt_info = json.load(gt_info_file)
        with open(self.sequence_path / 'scene_camera.json') as scene_camera_file:
            self.scene_camera = json.load(scene_camera_file)


        # object list
        with open('./datasets/ycb_video_classes.txt', 'r') as class_name_file:
            self.object_name_list = class_name_file.read().split('\n')

        self.obj_id = self.scene_gt['0'][self.prediction_id]['obj_id']
        self.object_id_str = f'obj_{self.obj_id:06d}'

        # FIXME arg?
        self.cosypose_results_path = cosypose_results_path

        # read CosyPose detections and predictions
        cosypose_results_path = Path(self.cosypose_results_path) / 'dataset=synpick' /  'results.pth.tar'
        # import ipdb; ipdb.set_trace()
        cosypose_results = torch.load(cosypose_results_path)['predictions']
        self.cosypose_bbox_detections = cosypose_results['maskrcnn_detections/detections']
        self.cosypose_pose_predictions = cosypose_results['maskrcnn_detections/refiner/iteration=4']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, depth, pose, intrinsics, mask = self.load(idx)

        image = torch.from_numpy(image).float()/255.0
        depth = torch.from_numpy(depth)
        mask = torch.from_numpy(mask)

        instance_mask = torch.zeros(3 * self.class_model_number)
        instance_mask[self.class_ids[0]*3 : self.class_ids[0]*3 + 3] = 1
        class_mask = (instance_mask==1)

        # check if this frame is keyframe
        obj_id = self.object_id_str
        D = self.cosypose_bbox_detections.infos
         detection_idx = D.loc[(D['scene_id'] == int(self.sequence_id)) & (D['view_id'] == int(idx)) & (D['label'] == obj_id)].index[0]
        # FIXME handle multiple instaces

        # use posecnn results for initialization
        center = np.array([0, 0])
        z = 0
        t_est = np.array([0, 0, 0], dtype=np.float32)
        q_est = np.array([1, 0, 0, 0], dtype=np.float32)
        
        roi = self.cosypose_bbox_detections.bboxes[detection_idx]
        pose = self.cosypose_pose_predictions.poses[detection_idx]

        pose_transform = Transform(pose.numpy())


        center[0] = (roi[0] + roi[2]) / 2.
        center[1] = (roi[1] + roi[3]) / 2
        z = pose[2, 3]
        t_est = pose[:3, 3].numpy()
        q_est = pose_transform.quaternion
        
        is_kf = False
        return image, depth, pose, intrinsics, class_mask, center, z, t_est, q_est, mask

    def load(self, idx):

        scene_id_str = f'{int(idx):06d}'
        class_id_str = f'{int(self.class_ids[0]):06d}'


        depth_file = self.sequence_path / 'depth' / f'{scene_id_str}.png'
        rgb_file = self.sequence_path / 'rgb' / f'{scene_id_str}.jpg'

        annotation = self.scene_gt[str(idx)]
        visib = self.scene_gt_info[str(idx)]

        cam_annotation = self.scene_camera[str(idx)]
        intrinsics = np.array(cam_annotation['cam_K']).reshape(3,3)

        scene_class_ids = [x['obj_id'] for x in annotation]
        img = np.array(Image.open(rgb_file))
        # import ipdb; ipdb.set_trace()

        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for _i, class_id in enumerate(scene_class_ids):
            print ("i , class_id", _i, class_id)
            mask_path = self.sequence_path / 'mask_visib' / f'{scene_id_str}_{int(_i):06d}.png'
            mask_n = np.array(Image.open(mask_path))
            mask[mask_n == 255] = class_id
        mask = np.expand_dims(mask, 2)

        depth = np.array(load_depth(depth_file))
        
        # element = [element for element in self.scene_gt[idx] if element['obj_id'] == self.class_ids[0]]
        # assert len(element) == 1, 'Only single instances supported'

        RCO = np.array(annotation[self.prediction_id]['cam_R_m2c']).reshape(3, 3)
        tCO = np.array(annotation[self.prediction_id]['cam_t_m2c']) * 0.001
        TC0 = Transform(RCO, tCO)
        # T0C = TC0.inverse()
        # T0O = T0C * TC0
        TC0 = TC0.toHomogeneousMatrix()
        pose = TC0[:3] # 3x4

        return img, depth, pose, intrinsics, mask


if __name__ == '__main__':
    target_obj = '002_master_chef_can'
    dataset_test = synpick_dataset(class_ids=[0],
                                    object_names=[target_obj],
                                    class_model_num=1,
                                    dataset_path='/home/user/periyasa/workspace/PoseRBPF/local_data/bop_datasets/synpick/train_synt',
                                    sequence_id='000003',
                                    prediction_id=2,
                                    cosypose_results_path='/home/user/periyasa/workspace/PoseRBPF/local_data/results/synpick--851320')
    image, depth, pose, intrinsics, class_mask, center, z, t_est, q_est, mask = dataset_test[0]


    print ('Alles Gut!!!')
