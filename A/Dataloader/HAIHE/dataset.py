import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import torch.nn.functional as F
from collections import Counter
class HAIHE(Dataset):
    def __init__(self, root_dir, labels_csv, feature_path, Labels_file, resize_size=(224, 224), augmentation=False, representation='rgb',oversample=True):
        """
        初始化 HAIHE 数据集类。
        
        参数：
            root_dir (str): 视频数据根目录路径。
            labels_csv (str): 标签 CSV 文件路径。
            feature_path (str): 特征文件根目录路径。
            label_map_path (str): 标签映射 JSON 文件路径。
            resize_size (tuple): 图像缩放的目标尺寸 (height, width)。
            augmentation (bool): 是否启用数据增强。
            representation (str): 事件图像的表示方式，可选 'rgb' 或 'gray_scale'。
        """
        self.root_dir = root_dir
        self.feature_path = feature_path
        self.resize_size = resize_size
        self.augmentation = augmentation
        self.representation = representation
        self.oversample = oversample
        
        # 读取标签映射
        with open(Labels_file, 'r') as f:
            self.label_map = json.load(f)
        
        # 读取 CSV 标签文件
        labels_df = pd.read_csv(labels_csv)
        self.labels_dict = dict(zip(labels_df['video_id'], labels_df['label']))
        
        # 获取视频目录列表
        self.video_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.video_dirs.sort()
        
        # 过滤掉没有标签的视频目录
        self.video_dirs = [d for d in self.video_dirs if self.get_video_id(d) in self.labels_dict]
        if len(self.video_dirs) == 0:
            raise ValueError("没有找到与标签 CSV 文件匹配的视频目录")
        if oversample:
            label_counts = Counter(self.labels_dict.values())
            max_count = max(label_counts.values())  # 多数类的样本数（normal）
            oversampled_dirs = []
            for video_dir in self.video_dirs:
                video_id = self.get_video_id(video_dir)
                label = self.labels_dict[video_id]
                # 计算需要复制的次数
                replication_factor = max_count // label_counts[label]
                oversampled_dirs.extend([video_dir] * replication_factor)
            self.video_dirs = oversampled_dirs
            print(f"过采样后样本数: {len(self.video_dirs)}, 各类别分布: {Counter([self.labels_dict[self.get_video_id(d)] for d in self.video_dirs])}")
    
    def get_video_id(self, video_dir):
        """
        从视频目录路径中提取 video_id。
        可以根据实际情况调整提取逻辑。
        """
        return os.path.basename(video_dir)
    
    def __len__(self):
        return len(self.video_dirs)
    
    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        video_id = self.get_video_id(video_dir)
        feature_dir = os.path.join(self.feature_path, video_id)
        
        if not os.path.exists(feature_dir):
            raise ValueError(f"特征目录不存在: {feature_dir}")
        
        # 加载特征文件，每三帧选一帧
        feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])
        if len(feature_files) == 0:
            raise ValueError(f"{feature_dir} 没有特征文件")
        selected_feature_files = feature_files[::5]
        frames_features = [torch.load(os.path.join(feature_dir, f)) for f in selected_feature_files]
        frames_features = torch.stack(frames_features, dim=0)
        
        # 加载事件帧，每三帧选一帧
        events_frames = self.load_frames(video_dir, 'event_frame', step=5)
        if len(events_frames) == 0:
            raise ValueError(f"{video_id} 的事件帧为空")
        
        # 调整帧数一致性
        min_frames = min(len(frames_features), len(events_frames))
        frames_features = frames_features[:min_frames]
        events_frames = events_frames[:min_frames]
        
        frame_lengths = len(events_frames)
        real_num_frame = frame_lengths
        
        events = self.process_images(events_frames, is_event=True)
        label_str = self.labels_dict[video_id]
        label = self.label_map[label_str]
        
        return frames_features.float(), torch.from_numpy(events).float(), label, frame_lengths, real_num_frame
    
    def load_frames(self, video_dir, prefix, step=5):
        """
        加载指定前缀的帧文件，并根据步长进行跳帧。
        """
        frame_files = sorted([f for f in os.listdir(video_dir) if f.startswith(prefix) and f.endswith(('.jpg', '.png'))])
        selected_frame_files = frame_files[::step]
        frames = [cv2.imread(os.path.join(video_dir, f)) for f in selected_frame_files]
        return frames
    
    def process_images(self, images, is_event=False):
        processed_images = [cv2.resize(img, self.resize_size) for img in images]
        
        if is_event and self.representation == 'gray_scale':
            processed_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in processed_images]
            processed_images = [np.stack([img, img, img], axis=-1) for img in processed_images]
        
        processed_images = np.stack(processed_images, axis=0)  # (T, H, W, C)
        processed_images = processed_images.transpose(0, 3, 1, 2)  # (T, C, H, W)
        processed_images = processed_images.astype(np.float32) / 255.0
        
        if self.augmentation:
            pass  # 可添加数据增强
        
        return processed_images

from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    # 提取批次中的各个元素
    video_features_batch = [item[0] for item in batch]  # 视频帧特征
    events_batch = [item[1] for item in batch]          # 事件数据
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)  # 标签
    frame_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)  # 帧长度
    real_num_frame = torch.tensor([item[4] for item in batch], dtype=torch.long)  # 实际帧数
    
    # 将 video_features 填充到相同帧数
    video_features = pad_sequence(video_features_batch, batch_first=True, padding_value=0)
    
    # 将 events 填充到相同帧数（如果需要）
    events = pad_sequence(events_batch, batch_first=True, padding_value=0)
    
    return video_features, events, labels, frame_lengths, real_num_frame