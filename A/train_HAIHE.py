import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os, gc, time, json
from os.path import join, abspath, dirname
import numpy as np
from model.EventCLIP import load_clip_to_cpu, EventCLIP
#from model.LossFunction import symmetric_cross_entropy_loss, symmetric_cross_entropy_loss_te_tv
from model.LossFunction import symmetric_cross_entropy_loss,symmetric_cross_entropy_loss_V
from model.utils.utils import read_yaml, seed_torch
import torch.nn as nn
import pandas as pd
from Dataloader.HAIHE.dataset import HAIHE, collate_fn
from collections import defaultdict, Counter
from utils.balanced_sampler import BalancedBatchSampler 

# 保存检查点的函数
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc1_ev, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc1_ev': best_acc1_ev,
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存至 {filepath}")

# 加载检查点的函数
def load_checkpoint(model, optimizer, scheduler, filepath):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location='cuda:0')  # 根据你的设备调整 map_location
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc1_ev = checkpoint['best_acc1_ev']
        print(f"从 {filepath} 加载检查点，将从 epoch {start_epoch} 开始训练")
        return start_epoch, best_acc1_ev
    else:
        print(f"在 {filepath} 未找到检查点，将从头开始训练")
        return 1, -np.inf  # 默认从 epoch 1 开始，最佳准确率为负无穷

def train_one_epoch(model, cfg, scaler, optimizer, scheduler, dataloader, epoch, weights):
    epoch_start = time.time()
    length = len(dataloader)
    running_loss, dataset_size, loss, epoch_loss = 0.0, 0.0, 0.0, 0.0
    classnames_num = len(json.load(open(cfg['Dataset']['Classnames'], 'r')))  # 5
    class_idxs = torch.arange(classnames_num, device=device)  # [0, 1, 2, 3, 4]
    for step, (frames_features, events, labels, frame_lengths, real_num_frame) in enumerate(dataloader):
        labels = labels.to(device)
        print(f"批次 {step}: labels: {labels.cpu().numpy()}, unique labels: {torch.unique(labels).cpu().numpy()}")
        print(f"批次 {step}: events 形状 {events.shape}, frames_features 形状 {frames_features.shape}, frame_lengths {frame_lengths}, 标签: {labels.unique().cpu().numpy()}")
        assert labels.max().item() < classnames_num, f"标签值 {labels.max().item()} 超出预期范围 [0, {classnames_num-1}]"
        batch_start = time.time()
        model = model.train().float()
        batch_size = frames_features.size(0)
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = events.half()
            frames_features = frames_features.half()
        with torch.cuda.amp.autocast(enabled=True):
            video_features, event_features, text_features_e, text_features_v \
                = model(events, frames_features, class_idxs, frame_lengths, real_num_frame)
            train_loss, mse_loss, loss_tim_im, logit_scale = 0.0, 0.0, 0.0, 100.0
            mse_loss = torch.tensor(mse_loss).to(video_features.device)
            loss_tim_im = torch.tensor(loss_tim_im).to(video_features.device)

            if cfg['LossFunction']['use_te_e']:
                print(f"********text_features_e shape before logits: {text_features_e.shape}")
                #logits_te_e = torch.einsum('bd,bcd->bc', event_features, text_features_e) * logit_scale
                logits_te_e = logit_scale * event_features @ text_features_e.t()
                print(f"logits_te_e shape: {logits_te_e.shape}, labels shape: {labels.shape}, weights shape: {weights.shape}")
                loss_te_e = symmetric_cross_entropy_loss_V(logits_te_e,labels)
                train_loss = train_loss + loss_te_e
            if cfg['LossFunction']['use_te_tim']:
                #logits_te_tv = torch.einsum('bnd,bmd->bnm', text_features_v, text_features_e) * logit_scale
                logits_te_tv = logit_scale * text_features_v @ text_features_e.t()
                print(f"logits_te_tv 形状: {logits_te_tv.shape}, labels 形状: {labels.shape}")
                loss_te_tv = symmetric_cross_entropy_loss(logits_te_tv)
                train_loss = train_loss + loss_te_tv
            if cfg['LossFunction']['use_im_ev_hi']:
                logit_v_ev_hi = logit_scale * event_features @ video_features.t()
                loss_v_ev_hi = symmetric_cross_entropy_loss(logit_v_ev_hi)
                train_loss = train_loss + loss_v_ev_hi

            if cfg['MODEL']['TextEncoder']['init_ctx'] and cfg['MODEL']['TextEncoder']['leranable_ctx']:
                mse_loss = torch.nn.MSELoss()(text_features_e, text_features_v)
                train_loss = train_loss + 0.1 * mse_loss

            loss_list = torch.stack([train_loss, loss_tim_im, loss_te_e, loss_te_tv, loss_v_ev_hi, mse_loss], dim=0) \
                        / cfg['Trainer']['accumulation_steps']

        scaler.scale(loss_list[0]).backward()
        if (step + 1) % cfg['Trainer']['accumulation_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += (loss_list.cpu().detach().numpy() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        batch_end = time.time()

        if (step) % cfg['Trainer']['print_freq'] == 0:
            if cfg['MODEL']['TextEncoder']['init_ctx'] and cfg['MODEL']['TextEncoder']['leranable_ctx']:
                print(
                    f'[{step + 1} / {length} | epoch: {epoch}] epoch_total_loss: {epoch_loss[0]:.7f} | '
                    f'epoch_tv_v_loss: {epoch_loss[1]:.7f} | '
                    f'epoch_te_ev_loss: {epoch_loss[2]:.7f} | '
                    f'epoch_te_tv_loss: {epoch_loss[3]:.7f} | '
                    f'epoch_v_ev_hi_loss: {epoch_loss[4]:.7f} | '
                    f'mse_loss: {epoch_loss[5]:.7f} | '
                    f'lr: {optimizer.param_groups[0]["lr"]:.7f} | '
                    f'batch_time: {(batch_end - batch_start):.3f} | '
                )
            else:
                print(
                    f'[{step + 1} / {length} | epoch: {epoch}] epoch_total_loss: {epoch_loss[0]:.7f} | '
                    f'epoch_tv_v_loss: {epoch_loss[1]:.7f} | '
                    f'epoch_te_ev_loss: {epoch_loss[2]:.7f} | '
                    f'epoch_te_tv_loss: {epoch_loss[3]:.7f} | '
                    f'epoch_v_ev_hi_loss: {epoch_loss[4]:.7f} | '
                    f'lr: {optimizer.param_groups[0]["lr"]:.7f} | '
                    f'batch_time: {(batch_end - batch_start):.3f} | '
                )

    scheduler.step()
    epoch_time = time.time() - epoch_start
    print(f"EPOCH {epoch} training takes {epoch_time}s.")
    gc.collect()
    return epoch_loss

def evaluate_one_epoch(model, cfg, dataloader, classnames_num, logit_scale):
    #classnames_idxs = np.arange(0, classnames_num)
    classnames_idxs = torch.arange(classnames_num, device=device)
    epoch_start = time.time()
    total, hit1_v, hit2_v, hit1_ev, hit2_ev = 0, 0, 0, 0, 0
    model = model.eval().float()

    class_correct_v = defaultdict(int)
    class_total_v = defaultdict(int)
    class_correct_ev = defaultdict(int)
    class_total_ev = defaultdict(int)
    
    all_logits_tv_v = []
    all_logits_te_e = []
    all_video_features = []
    all_event_features = []
    all_label = []

    all_labels = []
    for _, _, labels, _, _ in dataloader:
        all_labels.extend(labels.cpu().numpy())
    unique_labels = np.unique(all_labels)
    print(f'验证集中唯一标签: {unique_labels}')
    assert set(unique_labels) == set(range(classnames_num)), "标签与类别数不匹配"

    for frames_features, events, labels, frame_lengths, real_num_frame in dataloader:
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = events.float()
            frames_features = frames_features.float()
        with torch.no_grad():
            video_features, event_features, text_features_e, text_features_v \
                = model(events, frames_features, classnames_idxs, frame_lengths, real_num_frame)
            print(f"text_features_v shape: {text_features_v.shape}")
            print(f"text_features_e shape: {text_features_e.shape}")
            logits_te_e =  logit_scale * event_features @ text_features_e.t()
            logits_tv_v =  logit_scale * video_features @ text_features_v.t()
            scores_v = logits_tv_v.softmax(dim=-1)
            scores_ev = logits_te_e.softmax(dim=-1)
            all_logits_tv_v.append(logits_tv_v)
            all_logits_te_e.append(logits_te_e)
            all_video_features.append(video_features)
            all_event_features.append(event_features)

        B, _ = scores_v.size()
        for i in range(B):
            total += 1
            scores_v_i = scores_v[i]
            scores_ev_i = scores_ev[i]
            label_i = labels[i].item()
            all_label.append(label_i)

            pred_v = scores_v_i.argmax().item()
            class_total_v[label_i] += 1
            if pred_v == label_i:
                class_correct_v[label_i] += 1
                hit1_v += 1
            if label_i in scores_v_i.topk(2)[1].cpu().detach().numpy():
                hit2_v += 1

            pred_ev = scores_ev_i.argmax().item()
            class_total_ev[label_i] += 1
            if pred_ev == label_i:
                class_correct_ev[label_i] += 1
                hit1_ev += 1
            if label_i in scores_ev_i.topk(2)[1].cpu().detach().numpy():
                hit2_ev += 1

            acc1_v = hit1_v / total * 100.
            acc2_v = hit2_v / total * 100.
            acc1_ev = hit1_ev / total * 100.
            acc2_ev = hit2_ev / total * 100.

            if total % cfg['Trainer']['print_freq'] == 0:
                print(f'[Evaluation] num_samples: {total}  '
                      f'cumulative_acc1_v: {acc1_v:.2f}%  '
                      f'cumulative_acc2_v: {acc2_v:.2f}%  '
                      f'cumulative_acc1_ev: {acc1_ev:.2f}%  '
                      f'cumulative_acc2_ev: {acc2_ev:.2f}%  ')

    print("\n=== 分类准确率（按类别） ===")
    for cls in range(classnames_num):
        acc_v = class_correct_v[cls] / class_total_v[cls] * 100 if class_total_v[cls] > 0 else 0
        acc_ev = class_correct_ev[cls] / class_total_ev[cls] * 100 if class_total_ev[cls] > 0 else 0
        print(f'类别 {cls}: 视频模态 acc1_v = {acc_v:.2f}%, 事件模态 acc1_ev = {acc_ev:.2f}%')

    print(f'整体准确率: v_ev_top1={acc1_v:.2f}%, v_ev_top2={acc2_v:.2f}%, ev_top1={acc1_ev:.2f}%, ev_top2={acc2_ev:.2f}%')

    all_video_features = torch.cat(all_video_features, dim=0)
    all_event_features = torch.cat(all_event_features, dim=0)
    logits_v_ev = logit_scale * all_video_features @ all_event_features.t()
    scores_v_ev = logits_v_ev.softmax(dim=-1)

    all_logits_tv_v = torch.cat(all_logits_tv_v, dim=0)
    all_logits_te_e = torch.cat(all_logits_te_e, dim=0)
    scores_tv_v = all_logits_tv_v.t().softmax(dim=-1)
    scores_te_e = all_logits_te_e.t().softmax(dim=-1)
    N, n = scores_tv_v.t().size()
    all_label = np.array(all_label)
    print(f"N = {N}, n = {n}, classnames_num = {classnames_num}, len(classnames_idxs) = {len(classnames_idxs)}")
    class_retrieval_stats_v = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0, 
                                   'top1_idx': [], 'top2_idx': [], 'top3_idx': []} 
                               for i in range(classnames_num)}
    class_retrieval_stats_e = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0, 
                                   'top1_idx': [], 'top2_idx': [], 'top3_idx': []} 
                               for i in range(classnames_num)}

    for i in range(n):  # n = classnames_num)
        label_i = classnames_idxs[i]  # 当前类别的标签
        label_i = label_i.item()
        score_tv_v_i = scores_tv_v[i]  # 文本到视频的得分
        scores_te_e_i = scores_te_e[i]  # 文本到事件的得分

        topk_1_v = score_tv_v_i.topk(1)[1].cpu().detach().numpy()[0]  # Top-1 索引
        topk_2_v = score_tv_v_i.topk(2)[1].cpu().detach().numpy()     # Top-2 索引
        topk_3_v = score_tv_v_i.topk(3)[1].cpu().detach().numpy()     # Top-3 索引

        class_retrieval_stats_v[label_i]['top1_idx'] = [topk_1_v]
        class_retrieval_stats_v[label_i]['top2_idx'] = topk_2_v.tolist()
        class_retrieval_stats_v[label_i]['top3_idx'] = topk_3_v.tolist()

        if all_label[topk_1_v] == label_i:
            class_retrieval_stats_v[label_i]['top1_count'] += 1
        if label_i in [all_label[idx] for idx in topk_2_v]:
            class_retrieval_stats_v[label_i]['top2_count'] += 1
        if label_i in [all_label[idx] for idx in topk_3_v]:
            class_retrieval_stats_v[label_i]['top3_count'] += 1

        topk_1_e = scores_te_e_i.topk(1)[1].cpu().detach().numpy()[0]  # Top-1 索引
        topk_2_e = scores_te_e_i.topk(2)[1].cpu().detach().numpy()     # Top-2 索引
        topk_3_e = scores_te_e_i.topk(3)[1].cpu().detach().numpy()     # Top-3 索引

        class_retrieval_stats_e[label_i]['top1_idx'] = [topk_1_e]
        class_retrieval_stats_e[label_i]['top2_idx'] = topk_2_e.tolist()
        class_retrieval_stats_e[label_i]['top3_idx'] = topk_3_e.tolist()

        if all_label[topk_1_e] == label_i:
            class_retrieval_stats_e[label_i]['top1_count'] += 1
        if label_i in [all_label[idx] for idx in topk_2_e]:
            class_retrieval_stats_e[label_i]['top2_count'] += 1
        if label_i in [all_label[idx] for idx in topk_3_e]:
            class_retrieval_stats_e[label_i]['top3_count'] += 1

    for cls in range(classnames_num):
        acc_retrieval_1_v_cls = class_retrieval_stats_v[cls]['top1_count'] / 1 * 100.
        acc_retrieval_2_v_cls = class_retrieval_stats_v[cls]['top2_count'] / 1 * 100.
        acc_retrieval_3_v_cls = class_retrieval_stats_v[cls]['top3_count'] / 1 * 100.
        acc_retrieval_1_e_cls = class_retrieval_stats_e[cls]['top1_count'] / 1 * 100.
        acc_retrieval_2_e_cls = class_retrieval_stats_e[cls]['top2_count'] / 1 * 100.
        acc_retrieval_3_e_cls = class_retrieval_stats_e[cls]['top3_count'] / 1 * 100.

        print(f'类别 {cls}:')
        print(f'  视频检索: Top-1={acc_retrieval_1_v_cls:.2f}% (索引: {class_retrieval_stats_v[cls]["top1_idx"]}), '
              f'Top-2={acc_retrieval_2_v_cls:.2f}% (索引: {class_retrieval_stats_v[cls]["top2_idx"]}), '
              f'Top-3={acc_retrieval_3_v_cls:.2f}% (索引: {class_retrieval_stats_v[cls]["top3_idx"]})')
        print(f'  事件检索: Top-1={acc_retrieval_1_e_cls:.2f}% (索引: {class_retrieval_stats_e[cls]["top1_idx"]}), '
              f'Top-2={acc_retrieval_2_e_cls:.2f}% (索引: {class_retrieval_stats_e[cls]["top2_idx"]}), '
              f'Top-3={acc_retrieval_3_e_cls:.2f}% (索引: {class_retrieval_stats_e[cls]["top3_idx"]})')

    class_retrieval_stats_v_e = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0, 
                                     'top1_idx': [], 'top2_idx': [], 'top3_idx': [], 'total': 0} 
                                 for i in range(classnames_num)}

    for i in range(N):  # N 是视频样本总数
        label_i = all_label[i]  # 当前视频的标签
        class_retrieval_stats_v_e[label_i]['total'] += 1

        scores_v_ev_i = scores_v_ev[i]  # 视频到事件的得分
        topk_1_v_e = scores_v_ev_i.topk(1)[1].cpu().detach().numpy()[0]  # Top-1 索引
        topk_2_v_e = scores_v_ev_i.topk(2)[1].cpu().detach().numpy()     # Top-2 索引
        topk_3_v_e = scores_v_ev_i.topk(3)[1].cpu().detach().numpy()     # Top-3 索引

        class_retrieval_stats_v_e[label_i]['top1_idx'].append(topk_1_v_e)
        class_retrieval_stats_v_e[label_i]['top2_idx'].append(topk_2_v_e.tolist())
        class_retrieval_stats_v_e[label_i]['top3_idx'].append(topk_3_v_e.tolist())

        if all_label[topk_1_v_e] == label_i:
            class_retrieval_stats_v_e[label_i]['top1_count'] += 1
        if label_i in [all_label[idx] for idx in topk_2_v_e]:
            class_retrieval_stats_v_e[label_i]['top2_count'] += 1
        if label_i in [all_label[idx] for idx in topk_3_v_e]:
            class_retrieval_stats_v_e[label_i]['top3_count'] += 1

    for cls in range(classnames_num):
        total = class_retrieval_stats_v_e[cls]['total']
        if total > 0:
            acc_retrieval_1_v_e_cls = class_retrieval_stats_v_e[cls]['top1_count'] / total * 100.
            acc_retrieval_2_v_e_cls = class_retrieval_stats_v_e[cls]['top2_count'] / total * 100.
            acc_retrieval_3_v_e_cls = class_retrieval_stats_v_e[cls]['top3_count'] / total * 100.
        else:
            acc_retrieval_1_v_e_cls = 0
            acc_retrieval_2_v_e_cls = 0
            acc_retrieval_3_v_e_cls = 0

        print(f'类别 {cls}:')
        print(f'  视频到事件检索: Top-1={acc_retrieval_1_v_e_cls:.2f}% (索引: {class_retrieval_stats_v_e[cls]["top1_idx"]}), '
              f'Top-2={acc_retrieval_2_v_e_cls:.2f}% (索引: {class_retrieval_stats_v_e[cls]["top2_idx"]}), '
              f'Top-3={acc_retrieval_3_v_e_cls:.2f}% (索引: {class_retrieval_stats_v_e[cls]["top3_idx"]})')

    acc_retrival_1_v = sum(stats['top1_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrival_2_v = sum(stats['top2_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrival_3_v = sum(stats['top3_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrival_1_e = sum(stats['top1_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrival_2_e = sum(stats['top2_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrival_3_e = sum(stats['top3_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrival_1_v_e = sum(stats['top1_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100
    acc_retrival_2_v_e = sum(stats['top2_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100
    acc_retrival_3_v_e = sum(stats['top3_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100

    del all_logits_tv_v, all_logits_te_e, all_video_features, all_event_features
    torch.cuda.empty_cache()
    gc.collect()

    return acc1_v, acc2_v, acc1_ev, acc2_ev, \
           acc_retrival_1_e, acc_retrival_2_e, acc_retrival_3_e, \
           acc_retrival_1_v, acc_retrival_2_v, acc_retrival_3_v, \
           acc_retrival_1_v_e, acc_retrival_2_v_e, acc_retrival_3_v_e

if __name__ == '__main__':
    cfg = read_yaml('/home/username001/nyq/EventBind-master/Configs/HAIHE.yaml')
    THIS_DIR = abspath(dirname(__file__))
    RESULT_DIR = join(THIS_DIR, "Result")
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    EXP_DIR = join(RESULT_DIR, f"{cfg['Wandb']['exp_group_name']}-" + str(cfg['Wandb']['exp_num']))
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    seed_torch(cfg['Trainer']['seed'])
    tf = open(cfg['Dataset']['Classnames'], "r")
    classnames_dict = json.load(tf)
    classnames_list = [i for i in classnames_dict.keys()]
    classnames_num = len(classnames_list)
    print(f"类别数: {classnames_num}")
    
    

    train_dataset = HAIHE(cfg['Dataset']['Train']['Path'], 
                          labels_csv='/home/username001/nyq/EventBind-master/Dataloader/HAIHE/train_labels.csv',
                          feature_path='/home/username001/nyq/features/train',
                          Labels_file=cfg['Dataset']['Train']['Labels_file'],
                          resize_size=(cfg['Dataset']['resize_size']),
                          representation=cfg['Dataset']['Representation'],
                          augmentation=cfg['Dataset']['Train']['Augmentation'])
    label_counts = Counter([label for _, _, label, _, _ in train_dataset])
    total_samples = sum(label_counts.values())
    classnames_num = len(label_counts)  # 类别数量
    weights = [total_samples / (classnames_num * label_counts[i]) for i in range(classnames_num)]
    weights = torch.tensor(weights, dtype=torch.float32)

# 创建加权随机采样器
    labels = [label for _, _, label, _, _ in train_dataset]
    sampler = WeightedRandomSampler(weights=[weights[label] for label in labels], num_samples=len(labels), replacement=True)

# 使用采样器创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg['Dataset']['Train']['Batch_size'],
                          sampler=sampler, num_workers=4, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn)
    #train_loader = DataLoader(train_dataset, batch_size=cfg['Dataset']['Train']['Batch_size'],
      #                        shuffle=True, drop_last=True, num_workers=4,
      #                        prefetch_factor=2, pin_memory=True, collate_fn=collate_fn)

    val_dataset = HAIHE(cfg['Dataset']['Val']['Path'], 
                        labels_csv='/home/username001/nyq/EventBind-master/Dataloader/HAIHE/train_labels.csv',  # 需替换为实际验证集 CSV
                        feature_path='/home/username001/nyq/features/val',
                        Labels_file=cfg['Dataset']['Train']['Labels_file'],
                        resize_size=(cfg['Dataset']['resize_size']),
                        representation=cfg['Dataset']['Representation'],
                        augmentation=cfg['Dataset']['Val']['Augmentation'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['Dataset']['Val']['Batch_size'],
                            shuffle=False, drop_last=False, num_workers=4,
                            prefetch_factor=2, pin_memory=True, collate_fn=collate_fn)

    label_counts = Counter([label for _, _, label, _, _ in train_dataset])
    total_samples = sum(label_counts.values())
    weights = [total_samples / (len(label_counts) * label_counts[i]) for i in range(classnames_num)]
    weights = torch.tensor(weights, dtype=torch.float32)

    gpus = cfg['Trainer']['GPU_ids']
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)

    print(f"Loading CLIP (backbone: {cfg['MODEL']['BACKBONE']['Name']})")
    clip_model_v = load_clip_to_cpu(cfg)
    for name, param in clip_model_v.named_parameters():
        param.requires_grad = False
    clip_model_ev = load_clip_to_cpu(cfg)
    for name, param in clip_model_ev.named_parameters():
        if cfg['MODEL']['EventEncoder']['train_clip_backbone']:
            param.requires_grad = True
        else:
            param.requires_grad = False

    EventCLIP = EventCLIP(cfg, clip_model_v, clip_model_ev).to(device)
    EventCLIP = nn.DataParallel(EventCLIP, device_ids=gpus, output_device=gpus[0])
    if cfg['MODEL']['Load_Path'] != 'None':
        EventCLIP.load_state_dict(torch.load(cfg['MODEL']['Load_Path'], map_location='cuda:0'), strict=False)

    optimizer = torch.optim.AdamW(EventCLIP.parameters(), lr=float(cfg['Trainer']['lr']),
                                  weight_decay=float(cfg['Trainer']['weight_decay']))
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['Trainer']['epoch'], eta_min=float(cfg['Trainer']['min_lr']))
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)

    num_epochs = cfg['Trainer']['epoch']
    logit_scale = clip_model_v.logit_scale.exp()

    checkpoint_path = join(EXP_DIR, "checkpoint.pth")
    start_epoch, best_acc1_ev = load_checkpoint(EventCLIP, optimizer, lr_sched, checkpoint_path)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_loss = train_one_epoch(EventCLIP, cfg, loss_scaler, optimizer, lr_sched, train_loader, epoch, weights)
        acc1_v, acc2_v, acc1_ev, acc2_ev, \
        acc_retrival_1_e, acc_retrival_2_e, acc_retrival_3_e, \
        acc_retrival_1_v, acc_retrival_2_v, acc_retrival_3_v, \
        acc_retrival_1_v_e, acc_retrival_2_v_e, acc_retrival_3_v_e \
            = evaluate_one_epoch(EventCLIP, cfg, val_loader, classnames_num, logit_scale)
        
        save_checkpoint(EventCLIP, optimizer, lr_sched, epoch, best_acc1_ev, checkpoint_path)
        
        if acc1_ev >= best_acc1_ev:
            print(f"准确率提升 ({best_acc1_ev:0.4f}% ---> {acc1_ev:0.4f}%)")
            best_acc1_ev = acc1_ev
            PATH = join(EXP_DIR, f"best_im_ev_epoch{acc1_ev}.bin")
            torch.save(EventCLIP.state_dict(), PATH)
            print(f"最佳模型已保存至 {PATH}")