import os
import sys
import torch
import numpy as np

from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transforms import *
from dataset.base_dataset import BaseKITTIMono3DDataset

import cv2


DEFAULT_FILTER_CONFIG = {
    'min_height': 25,
    'min_depth': 2,
    'max_depth': 65,
    'max_truncation': 0.5,
    'max_occlusion': 2,
}


default_train_transforms = [
    PhotometricDistortion(
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    RandomShift(prob=0.5, shift_range=(-32, 32), hide_kpts_in_shift_area=True),
    RandomHorizontalFlip(prob=0.5),
    RandomCrop3D(prob=0.5, crop_size=(320, 960), hide_kpts_in_crop_area=True),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    Pad(size_divisor=32),
    ToTensor(),
]


default_test_transforms = [
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    Pad(size_divisor=32),
    ToTensor(),
]


class MonoConDataset(BaseKITTIMono3DDataset):
    def __init__(self, 
                 base_root: str, 
                 split: str,
                 max_objs: int = 30,
                 transforms: List[BaseTransform] = None,
                 filter_configs: Dict[str, Any] = None,
                 **kwargs):
        
        super().__init__(base_root=base_root, split=split, **kwargs)
        
        self.max_objs = max_objs
        
        if transforms is None:
            if (split == 'train'):
                transforms = default_train_transforms
            else:
                transforms = default_test_transforms
        self.transforms = Compose(transforms)
        
        if filter_configs is None:
            filter_configs = DEFAULT_FILTER_CONFIG
        else:
            cfg_keys = list(filter_configs.keys())
            flag = all([(key in DEFAULT_FILTER_CONFIG) for key in cfg_keys])
            assert flag, f"Keys in argument 'configs' must be one in {list(DEFAULT_FILTER_CONFIG.keys())}."
            
        for k, v in filter_configs.items():
            setattr(self, k, v)
        self.filter_configs = filter_configs
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        image, img_metas = self.load_image(idx)
        calib = self.load_calib(idx)
        

        raw_labels = []
        if self.split != 'test':
            # Raw State: Cam0 + Bottom-Center + Global Yaw
            # Converted to Cam2 + Local Yaw
            raw_labels = self.load_label(idx)
            raw_labels.convert_cam(src_cam=0, dst_cam=2)
            raw_labels.convert_yaw(src_type='global', dst_type='local')

        # Data Augmentation include
        if self.split == "train":
            image, raw_label = self.image_augmentation(image, raw_labels)
        
        new_labels = self._create_empty_labels()
        
        input_hw = image.shape[:2]
        for obj_idx, raw_label in enumerate(raw_labels):
            
            # Base Properties
            occ = raw_label.occlusion
            trunc = raw_label.truncation
            
            if (occ > self.max_occlusion) or (trunc > self.max_truncation):
                continue
            
            
            # 2D Box Properties
            gt_bbox = raw_label.box2d
            bbox_height = (gt_bbox[3] - gt_bbox[1])
            gt_label = raw_label.cls_num
            
            if bbox_height < self.min_height:
                continue
            
            
            # 3D Box Properties
            gt_bbox_3d = np.concatenate([
                raw_label.loc,
                raw_label.dim,
                np.array([raw_label.ry])
            ], axis=0)
            gt_label_3d = gt_label
            
            
            # 2D-3D Properties
            projected = raw_label.projected_center
            center2d, depth = projected[:-1], projected[-1]
            
            if not (self.min_depth <= depth <= self.max_depth):
                continue
            
            
            # 2D Keypoints
            keypoints = raw_label.projected_kpts            # (9, 3) / 8 Corners + 1 Center
            for k_idx, keypoint in enumerate(keypoints):
                kptx, kpty, _ = keypoint
                
                is_kpt_in_image = (0 <= kptx <= input_hw[1]) and (0 <= kpty <= input_hw[0])
                if is_kpt_in_image:
                    keypoints[k_idx, 2] = 2
            
            
            # Add Labels
            new_labels['gt_bboxes'][obj_idx, :] = gt_bbox
            new_labels['gt_labels'][obj_idx] = gt_label
            
            new_labels['gt_bboxes_3d'][obj_idx, :] = gt_bbox_3d
            new_labels['gt_labels_3d'][obj_idx] = gt_label_3d
            
            new_labels['centers2d'][obj_idx] = center2d
            new_labels['depths'][obj_idx] = depth
            
            new_labels['gt_kpts_2d'][obj_idx] = keypoints[:, :2].reshape(-1)
            new_labels['gt_kpts_valid_mask'][obj_idx] = keypoints[:, 2]
            
            new_labels['mask'][obj_idx] = True
        
        result_dict = {
            'img': image,
            'img_metas': img_metas,
            'calib': calib,
            'label': new_labels}
        
        result_dict = self.transforms(result_dict)
        return result_dict

    def image_augmentation(self, image, raw_labels):
        # Apply horizontal flip with a probability of 0.5
        # if np.random.rand() < 0.5:
        #     image = cv2.flip(image, 1)  # 1 denotes horizontal flip
        #     raw_labels.flip(image.shape[1])  # Update labels accordingly

        # Apply Gaussian noise
        # if self.split == 'train':
        #     mean = 0
        #     var = 10
        #     sigma = var ** 0.5
        #     gaussian = np.random.normal(mean, sigma, (image.shape[0],image.shape[1]))
        #     noisy_image = image + gaussian
            
            # image = GaussianNoise()(image=image)['image']


        if self.split == 'train':
            alpha = np.random.randint(0, 30) / 100  # Alpha controls the degree of grayness (0 = grayscale, 1 = original color)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Convert back to BGR format
            image = cv2.addWeighted(image, alpha, gray_image, 1 - alpha, 0)

            def adjust_brightness_contrast(img, brightness=0, contrast=0):
                if brightness != 0:
                    if brightness > 0:
                        shadow = brightness
                        highlight = 255
                    else:
                        shadow = 0
                        highlight = 255 + brightness
                    alpha_b = (highlight - shadow) / 255
                    gamma_b = shadow
                    
                    buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
                else:
                    buf = img.copy()

                if contrast != 0:
                    f = 131 * (contrast + 127) / (127 * (131 - contrast))
                    alpha_c = f
                    gamma_c = 127 * (1 - f)
                    
                    buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

                return buf
            # Apply fog effect
            brightness_para = np.random.randint(-150, 150)
            image = adjust_brightness_contrast(image, brightness=brightness_para, contrast=-50)

        return image, raw_labels
            
    def _create_empty_labels(self) -> Dict[str, np.ndarray]:
        annot_dict = {
            'gt_bboxes': np.zeros((self.max_objs, 4), dtype=np.float32),
            'gt_labels': np.zeros(self.max_objs, dtype=np.uint8),
            'gt_bboxes_3d': np.zeros((self.max_objs, 7), dtype=np.float32),
            'gt_labels_3d': np.zeros(self.max_objs, dtype=np.uint8),
            'centers2d': np.zeros((self.max_objs, 2), dtype=np.float32),
            'depths': np.zeros(self.max_objs, dtype=np.float32),
            'gt_kpts_2d': np.zeros((self.max_objs, 18), dtype=np.float32),
            'gt_kpts_valid_mask': np.zeros((self.max_objs, 9), dtype=np.uint8),
            'mask': np.zeros((self.max_objs,), dtype=np.bool_)}
        return annot_dict
    
    @staticmethod
    def collate_fn(batched: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Merge Image
        merged_image = torch.cat([d['img'].unsqueeze(0) for d in batched], dim=0)
        
        # Merge Image Metas
        img_metas_list = [d['img_metas'] for d in batched]
        merged_metas = {k: [] for k in img_metas_list[0].keys()}
        
        for img_metas in img_metas_list:
            for k, v in img_metas.items():
                merged_metas[k].append(v)
        
        # Merge Calib
        merged_calib = [d['calib'] for d in batched]
        
        # Merge Label
        label_list = [d['label'] for d in batched]
        
        label_keys = label_list[0].keys()
        merged_label = {k: None for k in label_keys}
        for key in label_keys:
            merged_label[key] = torch.cat([label[key] for label in label_list], dim=0)

        return {'img': merged_image, 
                'img_metas': merged_metas, 
                'calib': merged_calib, 
                'label': merged_label}
