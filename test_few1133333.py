import os
import argparse
import random
import math
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise, accuracy_score, f1_score, confusion_matrix, \
    matthews_corrcoef
from loss import FocalLoss, BinaryDiceLoss
from utils import cos_sim, encode_text_with_prompt_ensemble, augment
from prompt import REAL_NAME

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
# use_cuda = False
# os.environ["CUDA_VISIBLE_DEVICES"] = ''
device = torch.device("cuda:0" if use_cuda else "cpu")
# gen and cropgen modify medical_few to fakeimg and shot txt
CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}

# Your good/ungood definitions
# Your good/ungood definitions
good1 = 'img'
good2 = 'ocunormal'  # over 50 dataset
good3 = 'Healthy'  # open dataset
good4 = 'rpnormalnew'  # figure share rp
good5 = 'drnormal'  # dr?whats that？
good6 = 'bigimg'  # wch dataset
good8 = 'topnormalmodi'  # TOP modified
good9 = 'drdmenor'  # new dr/dme
good10 = 'mixnormal'  # validation
good11 = 'mmr'  # new mmr dataset
good = good10

CLASS_UNGOOD3 = ['AMD1', 'DR1', 'PM1', 'RD1', 'RVO1', 'Uveitis1']
CLASS_UNGOOD1 = ['RD', 'CRT', 'RB', 'PSD', 'LD', 'WWOP', 'OST', 'XGBX', 'PH', 'CWS', 'HD', 'HH']
CLASS_UNGOOD2 = ['ocu']
CLASS_UNGOOD4 = ['rp150']
CLASS_UNGOOD5 = ['drabnormal']
CLASS_UNGOOD6 = ['imgall3000']
CLASS_UNGOOD7 = ['newopen']
CLASS_UNGOOD8 = ['topabnormal']
CLASS_UNGOOD9 = ['drdmeabnor']
CLASS_UNGOOD10 = ['mixabnormal']
CLASS_UNGOOD11 = ['mmrab']
patt = '/home/liaogl/mvfa/chpngxinpao/32/Retina_OCT2017-103-0.8849.pth'
patt = '/home/liaogl/mvfa/32168933/Retina_OCT20178933.pth'  # 894175 8908jiushi8908
# patt = '/home/liaogl/mvfa/3216812162/Retina_OCT2017.pth' #8908
# patt = '/home/liaogl/mvfa/zheciduile/32nofake/Retina_OCT2017-145-0.8908.pth'
# patt = '/home/liaogl/mvfa/chpngxinpao/32/Retina_OCT2017-81-0.8857.pth'
patt = '/home/liaogl/mvfa/zheciduile/32nofake/Retina_OCT2017-145-0.8908.pth'
patt = '/home/liaogl/mvfa/zheciduile/32nofakede/Retina_OCT2017-34-0.8891.pth'
patt = '/home/liaogl/mvfa/laiba/32ex/Retina_OCT2017-95-0.8832.pth'
fls = [12, 16, 20]
print(good)
if good == good1 or good == good6:
    CLASS_UNGOOD = CLASS_UNGOOD6
elif good == good2:
    CLASS_UNGOOD = CLASS_UNGOOD2
elif good == good3:
    CLASS_UNGOOD = CLASS_UNGOOD7
elif good == good4:
    CLASS_UNGOOD = CLASS_UNGOOD4
elif good == good8:
    CLASS_UNGOOD = CLASS_UNGOOD8
elif good == good9:
    CLASS_UNGOOD = CLASS_UNGOOD9
elif good == good10:
    CLASS_UNGOOD = CLASS_UNGOOD10
elif good == good11:
    CLASS_UNGOOD = CLASS_UNGOOD11
else:
    CLASS_UNGOOD = CLASS_UNGOOD5


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 32 gen 32 71  32 crop 3 32 gencrop 38 32 ori 66.5
# 32leogenn 47 48 57 76 46 33 32
# 32leogen2 改了txt的几个生成图 改了knn3  99 98 97 96 95 94 73bes 77-80 99hao 96ok
# 32leogenzero 加回zero knn3  59 60 61 64 72 74
def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Retina_OCT2017')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument('--seed', type=int, default=111)
    # 测试改以下   检查shot datapath medifew txt
    parser.add_argument("--features_list", type=int, nargs="+", default=fls, help="features used")
    parser.add_argument('--save_path', type=str, default='./ckpt/32shot3216/')
    parser.add_argument('--data_path', type=str, default='./data/', help="data cropdata")
    parser.add_argument('--shot', type=int, default=32, help="shot")
    args = parser.parse_args()

    setup_seed(args.seed)

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                              pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list, lora_rank=32, lora_alpha=16,
                           lora_dropout=0.5).to(device)
    model.eval()

    checkpoint = torch.load(patt)
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    # det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    for cls_ungood in CLASS_UNGOOD:
        # for cls_ungood in CLASS_UNGOOD:
        # load test dataset
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        test_dataset = MedDataset(args.data_path, args.obj, cls_ungood, good, args.img_size, args.shot, args.iterate)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
        print(len(augment_normal_img))
        # memory bank construction
        support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
        support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.batch_size, shuffle=True,
                                                     **kwargs)
        # text prompt
        with torch.cuda.amp.autocast(), torch.no_grad():
            text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

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
        print(cls_ungood, '----------------------------')

        result, confu, ac = test(args, model, test_loader, text_features, det_mem_features)


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


def augment1(fewshot_img, fewshot_mask=None, aug_config=None):
    """
    Augment images with optional control over which augmentations to apply
    aug_config: dict with keys 'rotate', 'translate', 'flip' (all default to True)
    """
    if aug_config is None:
        # aug_config = {'rotate': True, 'translate': True, 'flip': True}
        aug_config = {'rotate': True, 'translate': True, 'flip': True, 'brightness': True, 'contrast': True,'noise': True}  # 这个对于auc有帮助 最好
        # aug_config =  {'rotate': False, 'translate': True, 'flip': True, 'brightness': True, 'contrast': True, 'noise': True} #这个对于auc有帮助
        # aug_config = {'rotate': False, 'translate': False, 'flip': True}

    B, C, H, W = fewshot_img.shape

    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        print('notnone')
        augment_fewshot_mask = fewshot_mask

        # rotate img
        if aug_config.get('rotate', True):
            for angle in [-np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, -np.pi / 32, np.pi / 32]:
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
        # rotate img
        print('nomask')
        if aug_config.get('rotate', True):
            for angle in [-np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, -np.pi / 32, np.pi / 32]:
            # for angle in [-np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8]:

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
            # flipped_img = vflip_img(flipped_img)
            # augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
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
            for scale in [1.05, 1.1]:
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


if __name__ == '__main__':
    main()


