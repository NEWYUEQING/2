import torch
from torch.utils.data import DataLoader
import os, gc, time, json
from os.path import join, abspath, dirname
import numpy as np
from model.EventCLIP import load_clip_to_cpu, EventCLIP
from model.utils.utils import read_yaml, seed_torch
import torch.nn as nn
import pandas as pd
from Dataloader.HAIHE.dataset import HAIHE, collate_fn
from collections import defaultdict, Counter
from utils.balanced_sampler import BalancedBatchSampler 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
def evaluate_one_epoch(model, cfg, dataloader, classnames_num, logit_scale):
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
    text_features_v_global = None
    first_batch = True

    all_labels = []
    for _, _, labels, _, _ in dataloader:
        all_labels.extend(labels.cpu().numpy())
    unique_labels = np.unique(all_labels)
    print(f'Unique labels in validation set: {unique_labels}')
    assert set(unique_labels) == set(range(classnames_num)), "Labels do not match the number of classes"

    for frames_features, events, labels, frame_lengths, real_num_frame in dataloader:
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = events.float()
            frames_features = frames_features.float()
        with torch.no_grad():
            video_features, event_features, text_features_e, text_features_v \
                = model(events, frames_features, classnames_idxs, frame_lengths, real_num_frame)
            if first_batch:
                text_features_v_global = text_features_v.detach().cpu()
                first_batch = False
            print(f"text_features_v shape: {text_features_v.shape}")
            print(f"text_features_e shape: {text_features_e.shape}")
            logits_te_e = logit_scale * event_features @ text_features_e.t()
            logits_tv_v = logit_scale * video_features @ text_features_v.t()
            scores_v = logits_tv_v.softmax(dim=-1)
            scores_ev = logits_te_e.softmax(dim=-1)
            all_logits_tv_v.append(logits_tv_v)
            all_logits_te_e.append(logits_te_e)
            all_video_features.append(video_features.cpu())
            all_event_features.append(event_features.cpu())

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

    print("\n=== Classification Accuracy (by class) ===")
    for cls in range(classnames_num):
        acc_v = class_correct_v[cls] / class_total_v[cls] * 100 if class_total_v[cls] > 0 else 0
        acc_ev = class_correct_ev[cls] / class_total_ev[cls] * 100 if class_total_ev[cls] > 0 else 0
        print(f'Class {cls}: Video modality acc1_v = {acc_v:.2f}%, Event modality acc1_ev = {acc_ev:.2f}%')

    print(f'Overall accuracy: v_ev_top1={acc1_v:.2f}%, v_ev_top2={acc2_v:.2f}%, ev_top1={acc1_ev:.2f}%, ev_top2={acc2_ev:.2f}%')

    all_video_features = torch.cat(all_video_features, dim=0)
    all_event_features = torch.cat(all_event_features, dim=0)
    all_label = np.array(all_label)
    logits_v_ev = logit_scale * all_video_features @ all_event_features.t()
    scores_v_ev = logits_v_ev.softmax(dim=-1)

    all_logits_tv_v = torch.cat(all_logits_tv_v, dim=0)
    all_logits_te_e = torch.cat(all_logits_te_e, dim=0)
    scores_tv_v = all_logits_tv_v.t().softmax(dim=-1)
    scores_te_e = all_logits_te_e.t().softmax(dim=-1)
    N, n = scores_tv_v.t().size()
    print(f"N = {N}, n = {n}, classnames_num = {classnames_num}, len(classnames_idxs) = {len(classnames_idxs)}")

    # Calculate retrieval accuracy
    class_retrieval_stats_v = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0, 
                                   'top1_idx': [], 'top2_idx': [], 'top3_idx': []} 
                               for i in range(classnames_num)}
    class_retrieval_stats_e = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0, 
                                   'top1_idx': [], 'top2_idx': [], 'top3_idx': []} 
                               for i in range(classnames_num)}

    for i in range(n):
        label_i = classnames_idxs[i].item()
        score_tv_v_i = scores_tv_v[i]
        scores_te_e_i = scores_te_e[i]

        topk_1_v = score_tv_v_i.topk(1)[1].cpu().detach().numpy()[0]
        topk_2_v = score_tv_v_i.topk(2)[1].cpu().detach().numpy()
        topk_3_v = score_tv_v_i.topk(3)[1].cpu().detach().numpy()

        class_retrieval_stats_v[label_i]['top1_idx'] = [topk_1_v]
        class_retrieval_stats_v[label_i]['top2_idx'] = topk_2_v.tolist()
        class_retrieval_stats_v[label_i]['top3_idx'] = topk_3_v.tolist()

        if all_label[topk_1_v] == label_i:
            class_retrieval_stats_v[label_i]['top1_count'] += 1
        if label_i in [all_label[idx] for idx in topk_2_v]:
            class_retrieval_stats_v[label_i]['top2_count'] += 1
        if label_i in [all_label[idx] for idx in topk_3_v]:
            class_retrieval_stats_v[label_i]['top3_count'] += 1

        topk_1_e = scores_te_e_i.topk(1)[1].cpu().detach().numpy()[0]
        topk_2_e = scores_te_e_i.topk(2)[1].cpu().detach().numpy()
        topk_3_e = scores_te_e_i.topk(3)[1].cpu().detach().numpy()

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

        print(f'Class {cls}:')
        print(f'  Video retrieval: Top-1={acc_retrieval_1_v_cls:.2f}% (indices: {class_retrieval_stats_v[cls]["top1_idx"]}), '
              f'Top-2={acc_retrieval_2_v_cls:.2f}% (indices: {class_retrieval_stats_v[cls]["top2_idx"]}), '
              f'Top-3={acc_retrieval_3_v_cls:.2f}% (indices: {class_retrieval_stats_v[cls]["top3_idx"]})')
        print(f'  Event retrieval: Top-1={acc_retrieval_1_e_cls:.2f}% (indices: {class_retrieval_stats_e[cls]["top1_idx"]}), '
              f'Top-2={acc_retrieval_2_e_cls:.2f}% (indices: {class_retrieval_stats_e[cls]["top2_idx"]}), '
              f'Top-3={acc_retrieval_3_e_cls:.2f}% (indices: {class_retrieval_stats_e[cls]["top3_idx"]})')

    class_retrieval_stats_v_e = {i: {'top1_count': 0, 'top2_count': 0, 'top3_count': 0, 
                                     'top1_idx': [], 'top2_idx': [], 'top3_idx': [], 'total': 0} 
                                 for i in range(classnames_num)}

    for i in range(N):
        label_i = all_label[i]
        class_retrieval_stats_v_e[label_i]['total'] += 1

        scores_v_ev_i = scores_v_ev[i]
        topk_1_v_e = scores_v_ev_i.topk(1)[1].cpu().detach().numpy()[0]
        topk_2_v_e = scores_v_ev_i.topk(2)[1].cpu().detach().numpy()
        topk_3_v_e = scores_v_ev_i.topk(3)[1].cpu().detach().numpy()

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

        print(f'Class {cls}:')
        print(f'  Video-to-event retrieval: Top-1={acc_retrieval_1_v_e_cls:.2f}% (indices: {class_retrieval_stats_v_e[cls]["top1_idx"]}), '
              f'Top-2={acc_retrieval_2_v_e_cls:.2f}% (indices: {class_retrieval_stats_v_e[cls]["top2_idx"]}), '
              f'Top-3={acc_retrieval_3_v_e_cls:.2f}% (indices: {class_retrieval_stats_v_e[cls]["top3_idx"]})')

    acc_retrieval_1_v = sum(stats['top1_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrieval_2_v = sum(stats['top2_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrieval_3_v = sum(stats['top3_count'] for stats in class_retrieval_stats_v.values()) / n * 100
    acc_retrieval_1_e = sum(stats['top1_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrieval_2_e = sum(stats['top2_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrieval_3_e = sum(stats['top3_count'] for stats in class_retrieval_stats_e.values()) / n * 100
    acc_retrieval_1_v_e = sum(stats['top1_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100
    acc_retrieval_2_v_e = sum(stats['top2_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100
    acc_retrieval_3_v_e = sum(stats['top3_count'] for stats in class_retrieval_stats_v_e.values()) / N * 100

    del all_logits_tv_v, all_logits_te_e
    torch.cuda.empty_cache()
    gc.collect()

    return acc1_v, acc2_v, acc1_ev, acc2_ev, \
           acc_retrieval_1_e, acc_retrieval_2_e, acc_retrieval_3_e, \
           acc_retrieval_1_v, acc_retrieval_2_v, acc_retrieval_3_v, \
           acc_retrieval_1_v_e, acc_retrieval_2_v_e, acc_retrieval_3_v_e, \
           all_video_features, all_event_features, text_features_v_global, all_label

if __name__ == '__main__':
    # Load configuration
    cfg = read_yaml('/home/username001/nyq/EventBind-master/Configs/HAIHE.yaml')

    # Set device
    gpus = cfg['Trainer']['GPU_ids']
    global device
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")

    # Load class names
    with open(cfg['Dataset']['Classnames'], 'r') as f:
        classnames_dict = json.load(f)
    classnames_num = len(classnames_dict)

    # Load CLIP model and get logit scale
    clip_model_v = load_clip_to_cpu(cfg)
    logit_scale = clip_model_v.logit_scale.exp()

    # Initialize EventCLIP model
    clip_model_ev = load_clip_to_cpu(cfg)
    EventCLIP = EventCLIP(cfg, clip_model_v, clip_model_ev).to(device)
    EventCLIP = nn.DataParallel(EventCLIP, device_ids=gpus, output_device=gpus[0])

    # Load pretrained weights if available
    if cfg['MODEL']['Load_Path'] != 'None':
        EventCLIP.load_state_dict(torch.load(cfg['MODEL']['Load_Path'], map_location='cuda:0'), strict=False)

    # Prepare validation dataset and dataloader
    val_dataset = HAIHE(cfg['Dataset']['Val']['Path'], 
                        labels_csv='/home/username001/nyq/EventBind-master/Dataloader/HAIHE/val_labels.csv',
                        feature_path='/home/username001/nyq/features/val',
                        Labels_file=cfg['Dataset']['Train']['Labels_file'],
                        resize_size=(cfg['Dataset']['resize_size']),
                        representation=cfg['Dataset']['Representation'],
                        augmentation=cfg['Dataset']['Val']['Augmentation'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['Dataset']['Val']['Batch_size'],
                            shuffle=False, drop_last=False, num_workers=4,
                            prefetch_factor=2, pin_memory=True, collate_fn=collate_fn)

    # Run evaluation
    results = evaluate_one_epoch(EventCLIP, cfg, val_loader, classnames_num, logit_scale)
    acc1_v, acc2_v, acc1_ev, acc2_ev, \
    acc_retrieval_1_e, acc_retrieval_2_e, acc_retrieval_3_e, \
    acc_retrieval_1_v, acc_retrieval_2_v, acc_retrieval_3_v, \
    acc_retrieval_1_v_e, acc_retrieval_2_v_e, acc_retrieval_3_v_e, \
    all_video_features, all_event_features, text_features_v_global, all_label = results

    # Print evaluation results
    print("\n=== Evaluation Results ===")
    print(f"Video modality Top-1 accuracy: {acc1_v:.2f}%")
    print(f"Video modality Top-2 accuracy: {acc2_v:.2f}%")
    print(f"Event modality Top-1 accuracy: {acc1_ev:.2f}%")
    print(f"Event modality Top-2 accuracy: {acc2_ev:.2f}%")
    print(f"Event retrieval Top-1: {acc_retrieval_1_e:.2f}%, Top-2: {acc_retrieval_2_e:.2f}%, Top-3: {acc_retrieval_3_e:.2f}%")
    print(f"Video retrieval Top-1: {acc_retrieval_1_v:.2f}%, Top-2: {acc_retrieval_2_v:.2f}%, Top-3: {acc_retrieval_3_v:.2f}%")
    print(f"Video-to-event retrieval Top-1: {acc_retrieval_1_v_e:.2f}%, Top-2: {acc_retrieval_2_v_e:.2f}%, Top-3: {acc_retrieval_3_v_e:.2f}%")

    # Generate feature space visualization
    num_samples = len(all_label)
    modality = np.array([0] * num_samples + [1] * num_samples + [2] * classnames_num)
    all_labels_plot = np.concatenate([all_label, all_label, np.arange(classnames_num)])
    all_features = torch.cat([all_video_features, all_event_features, text_features_v_global], dim=0).numpy()

    # Use t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    # Define shapes and colors
    shapes = ['o', 's', '^']  # Video: circle, Event: square, Text: triangle
    colors = matplotlib.colormaps['tab10']  # Updated to avoid deprecation warning

    # Plot scatter plot
    plt.figure(figsize=(10, 8))
    for mod in range(3):
        idx = (modality == mod)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                    c=all_labels_plot[idx], cmap=colors, marker=shapes[mod], 
                    label=['Video', 'Event', 'Text'][mod], alpha=0.6)

    plt.colorbar(label='Class')
    plt.legend()
    plt.title('Feature Space Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('/home/username001/nyq/results/feature_space_visualization1.png')
    plt.show()