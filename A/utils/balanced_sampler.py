import torch  
from torch.utils.data.sampler import Sampler  
import numpy as np  
  
class BalancedBatchSampler(Sampler):  
    """  
    采样器确保每个批次中一半是normal类别一半是其他类别  
    """  
    def __init__(self, dataset, batch_size, labels_dict):  
        self.dataset = dataset  
        if batch_size % 2 != 0:  
            raise ValueError("批次大小必须是偶数，以确保可以平均分配类别")  
        self.batch_size = batch_size  
        self.labels_dict = labels_dict  
          
        # 将样本索引按类别分组  
        self.normal_indices = []  
        self.other_indices = []  
          
        for i, video_dir in enumerate(dataset.video_dirs):  
            video_id = dataset.get_video_id(video_dir)  
            label = labels_dict[video_id]  
            if label == 'normal':  
                self.normal_indices.append(i)  
            else:  
                self.other_indices.append(i)  
          
        # 计算可以创建的完整批次数量  
        self.num_batches = min(len(self.normal_indices), len(self.other_indices)) * 2 // batch_size  
          
    def __iter__(self):  
        # 打乱每个类别的索引  
        np.random.shuffle(self.normal_indices)  
        np.random.shuffle(self.other_indices)  
          
        # 创建批次  
        for i in range(self.num_batches):  
            batch = []  
            # 添加normal类别样本  
            normal_count = self.batch_size // 2  
            normal_start = (i * normal_count) % len(self.normal_indices)  
            for j in range(normal_count):  
                idx = normal_start + j  
                if idx >= len(self.normal_indices):  
                    idx = idx % len(self.normal_indices)  
                batch.append(self.normal_indices[idx])  
              
            # 添加其他类别样本  
            other_count = self.batch_size // 2  
            other_start = (i * other_count) % len(self.other_indices)  
            for j in range(other_count):  
                idx = other_start + j  
                if idx >= len(self.other_indices):  
                    idx = idx % len(self.other_indices)  
                batch.append(self.other_indices[idx])  
              
            # 打乱批次中的样本顺序  
            np.random.shuffle(batch)  
            yield from batch  
      
    def __len__(self):  
        return self.num_batches * self.batch_size