# 这个脚本只有warmup，lrsch，aug的自适应没有生效，但是效果还可以
import os
import argparse
import random
import math
from pickle import FALSE
from torchvision import transforms

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise, accuracy_score, confusion_matrix, \
    matthews_corrcoef
from loss import FocalLoss, BinaryDiceLoss
from utils import  augment,cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import warnings

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#Retina_OCT2017 acc : 0.8681
# [[314  52]
#  [105 719]]
# 0.7058102746525584
# 0.9305798716112261  lr0001     005 711 002 6798
# Best result
# Retina_OCT2017 acc : 0.8714
# [[326  40]
#  [113 711]]
# 0.720739113146643
# 0.939973606026845
# Best result   duo aguRetina_OCT2017 acc : 0.884
# [[314  52]
#  [ 86 738]]
# 0.7361652489183076
# 0.9435845402939147
# Best result  /laiba/{kk}stepori

def main():
    parser = argparse.ArgumentParser(description='Testing')
    kk = 32
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Retina_OCT2017')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default=f'./laiba/{kk}stepori04/')
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 16,20], help="features used,121620best")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=kk)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=500, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--lr_schedule", type=str, default='step',
                        help="LR schedule: cosine, step, exponential, plateau, or none")
    parser.add_argument("--lr_warmup_epochs", type=int, default=60,
                        help="Number of warmup epochs")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum learning rate")

    parser.add_argument('--aug_schedule', type=str, default='full',
                        help='Augmentation schedule: full, progressive, adaptive, or none')


    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    print(args.features_list)

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                              pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    # create_medical_clip 7941 84 00
    model = CLIP_Inplanted(clip_model=clip_model, lora_rank=32, lora_alpha=16, lora_dropout=0.5,
                           features=args.features_list).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters我的识别码:167271173
    # 使用向日葵即可对我发起远程协助
    # 向日葵下载地址:http://url.oray.com/tGJdas/
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # Create LR scheduler
    lr_scheduler, warmup_scheduler = get_lr_scheduler(det_optimizer, args)

    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = MedDataset(args.data_path, args.obj, 'mixabnormal', 'mixnormal', args.img_size, args.shot,
                              args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # Create a pool of all available abnormal images for dynamic sampling
    abnormal_pool = create_abnormal_pool(args.data_path, args.obj, args.img_size)

    # Create normal images once (they remain constant)
    # Initial augmentation with full config for memory bank
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0
    loss_history = []  # Track loss for adaptive augmentation
    for epoch in range(args.epoch):
        print(f'Epoch {epoch}:')

        # Determine augmentation strategy for this epoch
        aug_config = get_augmentation_config(epoch, args.epoch, schedule=args.aug_schedule)
        print(f"Augmentation config: {aug_config}")
        abnum = args.shot
        k = 20
        ratio = 0.1 + 0.4 * (1 - math.exp(-epoch / k))

        # Dynamically sample abnormal images for this epoch
        sampled_abnormal_indices = random.sample(range(len(abnormal_pool)), min(abnum, len(abnormal_pool)))
        sampled_abnormal_images = [abnormal_pool[i] for i in sampled_abnormal_indices]

        # Convert to tensor and augment
        sampled_abnormal_tensor = torch.stack(sampled_abnormal_images)
        augment_abnorm_img, augment_abnorm_mask = augment(sampled_abnormal_tensor)

        # Re-augment normal images with current config
        augment_normal_img_epoch, augment_normal_mask_epoch = augment(test_dataset.fewshot_norm_img,
                                                                      )

        # Create training dataset for this epoch
        augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img_epoch], dim=0)
        augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask_epoch], dim=0)

        # FIX: Create labels based on actual lengths after augmentation
        augment_fewshot_label = torch.cat([
            torch.ones(len(augment_abnorm_img)),  # Abnormal labels
            torch.zeros(len(augment_normal_img_epoch))  # Normal labels
        ], dim=0)

        # Debug print to verify sizes match
        print(f"Augmented abnormal images: {len(augment_abnorm_img)}")
        print(f"Augmented normal images: {len(augment_normal_img_epoch)}")
        print(f"Total images: {len(augment_fewshot_img)}")
        print(f"Total labels: {len(augment_fewshot_label)}")
        print(f"Total masks: {len(augment_fewshot_mask)}")

        train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask,
                                                       augment_fewshot_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        # Training loop for this epoch
        loss_list = []
        correct_preds = 0
        total_preds = 0

        for (image, gt, label) in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
            image = image.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                _, det_patch_tokens = model(image)
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

                # det loss
                det_loss = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1,
                                                                                                     keepdim=True)
                    anomaly_map = 100.0 * det_patch_tokens[layer] @ text_features
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, label)

                loss = det_loss
                loss.requires_grad_(True)
                det_optimizer.zero_grad()
                loss.backward()
                det_optimizer.step()

                loss_list.append(loss.item())

                # Calculate training accuracy
                pred_label = (anomaly_score > 0.5).long()
                correct_preds += (pred_label == label).sum().item()
                total_preds += label.size(0)

        train_acc = correct_preds / total_preds * 100
        avg_loss = np.mean(loss_list)
        loss_history.append(avg_loss)

        # Update learning rate
        current_lr = update_lr_scheduler(lr_scheduler, warmup_scheduler, epoch,
                                         metric=None, warmup_epochs=args.lr_warmup_epochs)

        print(f"Loss: {avg_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        if current_lr is not None:
            print(f"Learning Rate: {current_lr:.6f}")
        print(f"Used {len(sampled_abnormal_images)} abnormal images this epoch")

        # Clear cache
        torch.cuda.empty_cache()

        # Build memory bank
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, det_patch_tokens = model(image)
                det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                det_features.append(det_patch_tokens)

        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in
                            range(len(det_features[0]))]

        result, confu, ac = test(args, model, test_loader, text_features, det_mem_features)

        # Update LR scheduler if using ReduceLROnPlateau
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            current_lr = update_lr_scheduler(lr_scheduler, warmup_scheduler, epoch,
                                             metric=result, warmup_epochs=args.lr_warmup_epochs)
            if current_lr is not None:
                print(f"Updated LR after validation: {current_lr:.6f}")

        ckp_path = os.path.join(args.save_path, f'{args.obj}-{epoch}-{ac}.pth')

        if result > best_result: # and confu[0][0] >= 300
            torch.save({'det_adapters': model.det_adapters.state_dict()}, ckp_path)
            best_result = result
            print("Best result\n")
    # for epoch in range(args.epoch):
    #     print(f'Epoch {epoch}:')
    #
    #     # Determine augmentation strategy for this epoch
    #     aug_config = get_augmentation_config(epoch, args.epoch, schedule=args.aug_schedule)
    #     print(f"Augmentation config: {aug_config}")
    #     abnum = args.shot
    #     k = 20
    #     ratio = 0.1 + 0.4 * (1 - math.exp(-epoch / k))
    #
    #     # Dynamically sample 32 abnormal images for this epoch
    #     sampled_abnormal_indices = random.sample(range(len(abnormal_pool)), min(kk, len(abnormal_pool)))
    #     sampled_abnormal_images = [abnormal_pool[i] for i in sampled_abnormal_indices]
    #
    #     # Convert to tensor and augment
    #     sampled_abnormal_tensor = torch.stack(sampled_abnormal_images)
    #     augment_abnorm_img, augment_abnorm_mask = augment(sampled_abnormal_tensor)
    #
    #     # Re-augment normal images with current config
    #     augment_normal_img_epoch, augment_normal_mask_epoch = augment(test_dataset.fewshot_norm_img,
    #                                                                   )
    #
    #     # Create training dataset for this epoch
    #     augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img_epoch], dim=0)
    #     augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask_epoch], dim=0)
    #     augment_fewshot_label = torch.cat([
    #         torch.Tensor([1] * len(augment_abnorm_img)),
    #         torch.Tensor([0] * len(augment_normal_img))
    #     ], dim=0)
    #
    #     train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask,
    #                                                    augment_fewshot_label)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    #
    #     # Training loop for this epoch
    #     loss_list = []
    #     correct_preds = 0
    #     total_preds = 0
    #
    #     for (image, gt, label) in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
    #         image = image.to(device)  # Shape: [batch_size, C, H, W]
    #         label = label.to(device)  # Shape: [batch_size]
    #
    #         with torch.cuda.amp.autocast():
    #             _, det_patch_tokens = model(image)
    #             # det_patch_tokens: list of [batch_size, num_patches+1, feat_dim]
    #             # Remove CLS token: [batch_size, num_patches, feat_dim]
    #             det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]
    #
    #             # det loss
    #             det_loss = 0
    #             for layer in range(len(det_patch_tokens)):
    #                 # Normalize features
    #                 det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1,
    #                                                                                                  keepdim=True)
    #
    #                 # Calculate anomaly map for entire batch
    #                 # Shape: [batch_size, num_patches, 2]
    #                 anomaly_map = 100.0 * det_patch_tokens[layer] @ text_features
    #                 anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]  # [batch_size, num_patches]
    #
    #                 # Calculate anomaly score per image
    #                 anomaly_score = torch.mean(anomaly_map, dim=-1)  # [batch_size]
    #                 det_loss += loss_bce(anomaly_score, label)
    #
    #             loss = det_loss
    #             loss.requires_grad_(True)
    #             det_optimizer.zero_grad()
    #             loss.backward()
    #             det_optimizer.step()
    #
    #             loss_list.append(loss.item())
    #
    #             # Calculate training accuracy
    #             pred_label = (anomaly_score > 0.5).long()
    #             correct_preds += (pred_label == label).sum().item()
    #             total_preds += label.size(0)
    #
    #     train_acc = correct_preds / total_preds * 100
    #     avg_loss = np.mean(loss_list)
    #     loss_history.append(avg_loss)
    #
    #     # Update learning rate
    #     current_lr = update_lr_scheduler(lr_scheduler, warmup_scheduler, epoch,
    #                                      metric=None, warmup_epochs=args.lr_warmup_epochs)
    #
    #     print(f"Loss: {avg_loss:.4f}")
    #     print(f"Training Accuracy: {train_acc:.2f}%")
    #     if current_lr is not None:
    #         print(f"Learning Rate: {current_lr:.6f}")
    #     print(f"Used {len(sampled_abnormal_images)} abnormal images this epoch")
    #
    #     # Clear cache
    #     torch.cuda.empty_cache()
    #
    #     # Build memory bank (keep batch_size=1 for memory bank)
    #     det_features = []
    #     for image in support_loader:
    #         image = image[0].to(device)
    #         with torch.no_grad():
    #             _, det_patch_tokens = model(image)
    #             det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
    #             det_features.append(det_patch_tokens)
    #
    #     det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in
    #                         range(len(det_features[0]))]
    #
    #     result, confu, ac = test(args, model, test_loader, text_features, det_mem_features)
    #
    #     # Update LR scheduler if using ReduceLROnPlateau
    #     if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #         current_lr = update_lr_scheduler(lr_scheduler, warmup_scheduler, epoch,
    #                                          metric=result, warmup_epochs=args.lr_warmup_epochs)
    #         if current_lr is not None:
    #             print(f"Updated LR after validation: {current_lr:.6f}")
    #
    #     ckp_path = os.path.join(args.save_path, f'{args.obj}-{epoch}-{ac}.pth')
    #
    #     if result > best_result and confu[0][0] >= 300:
    #         torch.save({'det_adapters': model.det_adapters.state_dict()}, ckp_path)
    #         best_result = result
    #         print("Best result\n")


def get_lr_scheduler(optimizer, args):
    """
    Create learning rate scheduler based on configuration

    Args:
        optimizer: PyTorch optimizer
        args: argument parser with lr_schedule, epoch, lr_warmup_epochs, lr_min

    Returns:
        scheduler object and warmup scheduler (if applicable)
    """
    if args.lr_schedule == 'cosine':
        # Cosine annealing: smooth decay from max to min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epoch - args.lr_warmup_epochs,
            eta_min=args.lr_min
        )
        print(f"Using Cosine Annealing LR: {args.learning_rate} -> {args.lr_min}")

    elif args.lr_schedule == 'step':
        # Step decay: drop LR at specific milestones
        milestones = [int(args.epoch * 0.3), int(args.epoch * 0.6), int(args.epoch * 0.8)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.5  # Multiply LR by 0.5 at each milestone
        )
        print(f"Using Step LR with milestones at epochs: {milestones}")

    elif args.lr_schedule == 'exponential':
        # Exponential decay: smooth continuous decay
        gamma = (args.lr_min / args.learning_rate) ** (1.0 / args.epoch)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        print(f"Using Exponential LR decay with gamma={gamma:.6f}")

    elif args.lr_schedule == 'plateau':
        # Reduce on plateau: adaptive based on validation performance
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # We want to maximize MCC/accuracy
            factor=0.5,
            patience=15,
            min_lr=args.lr_min
        )
        print("Using ReduceLROnPlateau scheduler (metric-based)")

    elif args.lr_schedule == 'none':
        scheduler = None
        print("No LR scheduling (constant learning rate)")

    else:
        raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")

    # Optional warmup scheduler
    warmup_scheduler = None
    if args.lr_warmup_epochs > 0 and args.lr_schedule != 'none':
        # Linear warmup from 0 to initial LR
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of initial LR
            end_factor=1.0,
            total_iters=args.lr_warmup_epochs
        )
        print(f"Using {args.lr_warmup_epochs} epochs of warmup")

    return scheduler, warmup_scheduler


def update_lr_scheduler(scheduler, warmup_scheduler, epoch, metric=None, warmup_epochs=0):
    """
    Update learning rate scheduler

    Args:
        scheduler: main LR scheduler
        warmup_scheduler: warmup scheduler (or None)
        epoch: current epoch
        metric: validation metric for ReduceLROnPlateau
        warmup_epochs: number of warmup epochs

    Returns:
        current learning rate
    """
    if epoch < warmup_epochs and warmup_scheduler is not None:
        warmup_scheduler.step()
        current_lr = warmup_scheduler.get_last_lr()[0]
    elif scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                scheduler.step(metric)
            current_lr = scheduler.optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
    else:
        current_lr = scheduler.optimizer.param_groups[0]['lr'] if scheduler else None

    return current_lr


def augment1(fewshot_img, fewshot_mask=None, aug_config=None):
    """
    Augment images with optional control over which augmentations to apply
    aug_config: dict with keys 'rotate', 'translate', 'flip' (all default to True)
    """
    if aug_config is None:
        aug_config = {'rotate': True, 'translate': True, 'flip': True}
        # aug_config =  {'rotate': True, 'translate': True, 'flip': True, 'brightness': True, 'contrast': True, 'noise': True} #这个对于auc有帮助 最好

    B, C, H, W = fewshot_img.shape

    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        augment_fewshot_mask = fewshot_mask

        # rotate img
        if aug_config.get('rotate', True):
            for angle in [-np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, -np.pi / 4, np.pi / 4]:
                from utils import rot_img
                rotate_img = rot_img(fewshot_img, angle)
                augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)
                rotate_mask = rot_img(fewshot_mask, angle)
                augment_fewshot_mask = torch.cat([augment_fewshot_mask, rotate_mask], dim=0)

        # translate img
        if aug_config.get('translate', True):
            for a, b in [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]:
                from utils import translation_img
                trans_img = translation_img(fewshot_img, a, b)
                augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)
                trans_mask = translation_img(fewshot_mask, a, b)
                augment_fewshot_mask = torch.cat([augment_fewshot_mask, trans_mask], dim=0)

        # flip img
        if aug_config.get('flip', True):
            from utils import hflip_img, vflip_img
            # hflip
            flipped_img = hflip_img(fewshot_img)
            augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
            flipped_mask = hflip_img(fewshot_mask)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)
            # vflip
            flipped_img = vflip_img(fewshot_img)
            augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
            flipped_mask = vflip_img(fewshot_mask)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

    else:
        print('nomask')
        # rotate img
        if aug_config.get('rotate', True):
            for angle in [-np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, -np.pi / 32, np.pi / 32]:
                from utils import rot_img
                rotate_img = rot_img(fewshot_img, angle)
                augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

        # translate img
        if aug_config.get('translate', True):
            for a, b in [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]:
                from utils import translation_img
                trans_img = translation_img(fewshot_img, a, b)
                augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

        # flip img
        if aug_config.get('flip', True):
            from utils import hflip_img, vflip_img
            # hflip
            flipped_img = hflip_img(fewshot_img)
            augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
            # vflip
            flipped_img = vflip_img(fewshot_img)
            augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
        if aug_config.get('brightness', False):
            for factor in [0.8, 1.2]:
                bright_img = torch.clamp(fewshot_img * factor, 0, 1)
                augment_fewshot_img = torch.cat([augment_fewshot_img, bright_img], dim=0)

        if aug_config.get('contrast', False):
            for factor in [0.8, 1.2]:
                mean = fewshot_img.mean(dim=(2, 3), keepdim=True)
                contrast_img = torch.clamp((fewshot_img - mean) * factor + mean, 0, 1)
                augment_fewshot_img = torch.cat([augment_fewshot_img, contrast_img], dim=0)

        if aug_config.get('noise', False):
            for std in [0.01, 0.02]:
                noise = torch.randn_like(fewshot_img) * std
                noisy_img = torch.clamp(fewshot_img + noise, 0, 1)
                augment_fewshot_img = torch.cat([augment_fewshot_img, noisy_img], dim=0)

        if aug_config.get('blur', False):
            kernel_size = 3
            blur_img = F.avg_pool2d(fewshot_img, kernel_size, stride=1, padding=kernel_size // 2)
            augment_fewshot_img = torch.cat([augment_fewshot_img, blur_img], dim=0)

        if aug_config.get('cutout', False):
            for _ in range(2):
                cutout_img = fewshot_img.clone()
                x = torch.randint(0, W - W // 8, (B,))
                y = torch.randint(0, H - H // 8, (B,))
                w, h = W // 8, H // 8
                for i in range(B):
                    cutout_img[i, :, y[i]:y[i] + h, x[i]:x[i] + w] = 0
                augment_fewshot_img = torch.cat([augment_fewshot_img, cutout_img], dim=0)

        if aug_config.get('color_jitter', False):
            for hue_shift in [-0.05, 0.05]:
                color_img = fewshot_img.clone()
                if C == 3:
                    r, g, b = color_img[:, 0], color_img[:, 1], color_img[:, 2]
                    if hue_shift > 0:
                        color_img = torch.stack([
                            torch.clamp(r * 1.05 + g * 0.05, 0, 1),
                            torch.clamp(g * 0.95, 0, 1), b
                        ], dim=1)
                    else:
                        color_img = torch.stack([
                            torch.clamp(r * 0.95, 0, 1),
                            torch.clamp(g * 1.05 + r * 0.05, 0, 1), b
                        ], dim=1)
                    augment_fewshot_img = torch.cat([augment_fewshot_img, color_img], dim=0)

        if aug_config.get('gamma', False):
            for gamma in [0.8, 1.2]:
                gamma_img = torch.clamp(fewshot_img ** gamma, 0, 1)
                augment_fewshot_img = torch.cat([augment_fewshot_img, gamma_img], dim=0)
        if aug_config.get('zoom', False):
            for scale in [0.9, 1.1]:
                if scale < 1.0:
                    # Zoom out: shrink image and pad
                    new_h, new_w = int(H * scale), int(W * scale)
                    zoomed = F.interpolate(fewshot_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    pad_h, pad_w = (H - new_h) // 2, (W - new_w) // 2
                    zoomed_img = F.pad(zoomed, (pad_w, W - new_w - pad_w, pad_h, H - new_h - pad_h), value=0)
                    augment_fewshot_img = torch.cat([augment_fewshot_img, zoomed_img], dim=0)

                else:
                    # Zoom in: enlarge and center crop
                    new_h, new_w = int(H * scale), int(W * scale)
                    zoomed = F.interpolate(fewshot_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    crop_h, crop_w = (new_h - H) // 2, (new_w - W) // 2
                    zoomed_img = zoomed[:, :, crop_h:crop_h + H, crop_w:crop_w + W]
                    augment_fewshot_img = torch.cat([augment_fewshot_img, zoomed_img], dim=0)

        B, _, H, W = augment_fewshot_img.shape
        augment_fewshot_mask = torch.zeros([B, 1, H, W])

    return augment_fewshot_img, augment_fewshot_mask


def get_augmentation_config(epoch, total_epochs, schedule='progressive', train_acc=None):
    """
    Determine which augmentations to apply based on training progress

    Args:
        epoch: current epoch
        total_epochs: total number of epochs
        schedule: 'full', 'progressive', 'adaptive', or 'none'
        train_acc: training accuracy for adaptive scheduling

    Returns:
        dict with augmentation flags
    """
    if schedule == 'none':
        return {'rotate': False, 'translate': False, 'flip': False}

    elif schedule == 'full':
        return  {'rotate': True, 'translate': True, 'flip': True, 'brightness': True, 'contrast': True, 'noise': True} #这个对于auc有帮助 最好
        # return {'rotate': True, 'translate': True, 'flip': True}

    elif schedule == 'progressive':
        progress = epoch / total_epochs
        if progress < 0.3:  # First 30%: Full augmentation
            return {'rotate': True, 'translate': True, 'flip': True}
        elif progress < 0.6:  # 30-60%: Medium augmentation
            return {'rotate': True, 'translate': False, 'flip': True}
        elif progress < 0.8:  # 60-80%: Light augmentation
            return {'rotate': False, 'translate': False, 'flip': True}
        else:  # Final 20%: Minimal augmentation
            return {'rotate': False, 'translate': False, 'flip': False}

    elif schedule == 'adaptive':
        # Based on training accuracy
        if train_acc is None:
            return {'rotate': True, 'translate': True, 'flip': True}

        if train_acc < 85:
            return {'rotate': True, 'translate': True, 'flip': True}
        elif train_acc < 92:
            return {'rotate': True, 'translate': False, 'flip': True}
        elif train_acc < 96:
            return {'rotate': False, 'translate': False, 'flip': True}
        else:
            return {'rotate': False, 'translate': False, 'flip': False}

    else:
        return {'rotate': True, 'translate': True, 'flip': True}
    """Create a pool of all available abnormal images for dynamic sampling"""
    # transform_x = transforms.Compose([
    #     transforms.Resize((img_size, img_size), Image.BICUBIC),
    #     transforms.ToTensor(),
    # ])
    #
    # abnormal_pool = []
    # dataset_path = os.path.join(data_path, f'{obj}_AD')
    # img_dir = os.path.join(dataset_path, 'valid', 'Ungood', 'fakeimg')
    #
    # if not os.path.exists(img_dir):
    #     print(f"Warning: Abnormal image directory {img_dir} does not exist")
    #     return abnormal_pool
    #
    # abnormal_names = os.listdir(img_dir)
    #
    # for f in abnormal_names:
    #     if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.tif'):
    #         image_path = os.path.join(img_dir, f)
    #         try:
    #             image = Image.open(image_path).convert('RGB')
    #             image = transform_x(image)
    #             abnormal_pool.append(image)
    #         except Exception as e:
    #             print(f"Error loading image {image_path}: {e}")
    #             continue
    #
    # print(f"Created abnormal pool with {len(abnormal_pool)} images")
    # return abnormal_pool


def test(args, model, test_loader, text_features, det_mem_features):
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []

    for (image, y, mask) in tqdm(test_loader, desc='Test'):
        image = image.to(device)  # Shape: [batch_size, C, H, W]
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, det_patch_tokens = model(image)
            # det_patch_tokens: list of [batch_size, num_patches+1, feat_dim]
            det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

            batch_size = image.shape[0]

            # Process each image in the batch
            for b in range(batch_size):
                # Extract features for single image
                single_image_tokens = [p[b] for p in det_patch_tokens]  # [num_patches, feat_dim]

                # Few-shot detection
                anomaly_maps_few_shot = []
                for idx, p in enumerate(single_image_tokens):
                    k = 3  # Number of nearest neighbors
                    cos = cos_sim(det_mem_features[idx], p)  # [mem_patches, test_patches]
                    height = int(np.sqrt(cos.shape[1]))

                    # Sort distances and take average of top-k
                    sorted_cos = torch.sort(1 - cos, dim=0).values
                    anomaly_map_few_shot = torch.mean(sorted_cos[:k, :], dim=0).reshape(1, 1, height, height)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())

                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)

                # Apply Gaussian filter
                anomaly_map_few_shot = np.array(anomaly_map_few_shot, dtype=np.float32)
                anomaly_map_few_shot = gaussian_filter(anomaly_map_few_shot, sigma=3)
                score_few_det = anomaly_map_few_shot.max()
                det_image_scores_few.append(score_few_det)

                # Zero-shot detection
                anomaly_score = 0
                for layer in range(len(single_image_tokens)):
                    normalized_tokens = single_image_tokens[layer] / single_image_tokens[layer].norm(dim=-1,
                                                                                                     keepdim=True)
                    anomaly_map = 100.0 * normalized_tokens @ text_features
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, 1]
                    anomaly_score += anomaly_map.max()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

        # Store ground truth for entire batch
        gt_mask_list.extend(mask.squeeze(1).cpu().detach().numpy())  # Handle batch dimension
        gt_list.extend(y.cpu().detach().numpy())

        # Clear cache
        torch.cuda.empty_cache()

    gt_list = np.array(gt_list)
    det_image_scores_zero = np.array(det_image_scores_zero)
    det_image_scores_few = np.array(det_image_scores_few)

    # Normalize scores
    det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (
            det_image_scores_zero.max() - det_image_scores_zero.min())
    det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (
            det_image_scores_few.max() - det_image_scores_few.min())

    image_scores = 0.3 * det_image_scores_zero + 0.7 * det_image_scores_few

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    image_scores = np.round(image_scores).astype(int)
    acc = accuracy_score(gt_list, image_scores)
    mcc = matthews_corrcoef(gt_list, image_scores)
    ac = round(acc, 4)
    confusion = confusion_matrix(gt_list, image_scores)

    print(f'{args.obj} acc : {round(acc, 4)}')
    print(confusion)
    print(mcc)
    print(img_roc_auc_det)

    return mcc, confusion, ac
def create_abnormal_pool(data_path, obj, img_size):
    """Create a pool of all available abnormal images for dynamic sampling"""
    transform_x = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
    ])

    abnormal_pool = []
    dataset_path = os.path.join(data_path, f'{obj}_AD')
    img_dir = os.path.join(dataset_path, 'valid', 'Ungood', 'fakeimg') #zheligai fake

    if not os.path.exists(img_dir):
        print(f"Warning: Abnormal image directory {img_dir} does not exist")
        return abnormal_pool

    abnormal_names = os.listdir(img_dir)

    for f in abnormal_names:
        if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.tif'):
            image_path = os.path.join(img_dir, f)
            try:
                image = Image.open(image_path).convert('RGB')
                image = transform_x(image)
                abnormal_pool.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

    print(f"Created abnormal pool with {len(abnormal_pool)} images")
    return abnormal_pool


if __name__ == '__main__':
    main()