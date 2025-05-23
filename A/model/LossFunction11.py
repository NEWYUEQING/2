import torch.nn.functional as F
import torch

def symmetric_cross_entropy_loss(logits, labels, weights):
    # 计算正向交叉熵损失
    loss_i = F.cross_entropy(logits, labels, weight=weights)
    # 这里不应使用 logits.T，因为它改变了 batch_size 的维度
    # 如果需要对称损失，可以考虑其他方式，例如对调特征计算
    # 但在此处，我们假设对称损失是对原始 logits 的某种变体
    loss_t = F.cross_entropy(logits, labels, weight=weights)  # 暂时与 loss_i 相同
    loss = (loss_i + loss_t) / 2
    return loss
def symmetric_cross_entropy_loss_te_tv(logits_te_tv, labels, weights):
    """
    计算对称交叉熵损失，用于 [batch_size, n_cls, n_cls] 形状的 logits。
    
    参数：
        logits_te_tv (torch.Tensor): 形状为 [batch_size, n_cls, n_cls] 的相似性矩阵
        labels (torch.Tensor): 形状为 [batch_size] 的真实标签
        weights (torch.Tensor): 形状为 [n_cls] 的类别权重
    
    返回：
        loss (torch.Tensor): 对称交叉熵损失
    """
    batch_size = logits_te_tv.size(0)
    
    # 计算行损失
    row_logits = logits_te_tv[torch.arange(batch_size), labels, :]  # 形状：[batch_size, n_cls]
    loss_row = F.cross_entropy(row_logits, labels, weight=weights)
    
    # 计算列损失
    col_logits = logits_te_tv.permute(0, 2, 1)[torch.arange(batch_size), labels, :]  # 形状：[batch_size, n_cls]
    loss_col = F.cross_entropy(col_logits, labels, weight=weights)
    
    # 对称损失：平均行和列损失
    loss = (loss_row + loss_col) / 2
    return loss