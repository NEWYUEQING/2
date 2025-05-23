import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=2)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class VideoEncoder(nn.Module):
    def __init__(self, feature_dim, conv_type=2, if_cls_token=True, use_middle_cls_token=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token

        # 可学习的CLS token
        if self.if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            trunc_normal_(self.cls_token, std=0.02)

        # TemporalConv模块
        self.temporal_conv = TemporalConv(
            input_size=feature_dim,
            hidden_size=feature_dim,  # 保持输入输出维度一致
            conv_type=conv_type
        )

        # 归一化层
        self.norm_f = nn.LayerNorm(feature_dim, eps=1e-5)

    def forward(self, frames_features):
        # 输入: [batch, num_frames, feature_dim]
        frames_features = frames_features.squeeze(2)
        batch_size, num_frames, _ = frames_features.shape

        # 添加CLS token（可选）
        if self.if_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, feature_dim]
            if self.use_middle_cls_token:
                token_position = num_frames // 2
                x = torch.cat([frames_features[:, :token_position, :], cls_token, frames_features[:, token_position:, :]], dim=1)
            else:
                x = torch.cat([cls_token, frames_features], dim=1)  # CLS在开头
                token_position = 0
        else:
            x = frames_features

        # 通过TemporalConv处理序列
        hidden_states = self.temporal_conv(x)  # [batch, num_frames or num_frames+1, feature_dim]

        # 归一化
        hidden_states = self.norm_f(hidden_states)

        # 提取视频级别特征
        if self.if_cls_token:
            video_features = hidden_states[:, token_position, :]  # 提取CLS token
        else:
            video_features = hidden_states.mean(dim=1)  # 平均池化
        print(f"Video features shape: {video_features.shape}")
        return video_features

# 使用示例
if __name__ == "__main__":
    # 假设你的帧特征
    batch_size, num_frames, feature_dim = 2, 10, 512
    frames_features = torch.randn(batch_size, num_frames, feature_dim)

    # 初始化模型
    model = VideoEncoder(feature_dim=feature_dim, conv_type=2, if_cls_token=True, use_middle_cls_token=False)

    # 前向传播
    video_features = model(frames_features)
    print(f"Video features shape: {video_features.shape}")  # 输出: [batch_size, feature_dim]