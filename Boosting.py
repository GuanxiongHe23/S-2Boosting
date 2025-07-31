import os
import argparse
import cv2
import random
import colorsys
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import numpy as np
import utils
import models
import time

from PIL import Image
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image
from loader import ImageFolderInstance
from tqdm import tqdm

from torch import optim
from torchvision.utils import save_image
from torchvision import transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from timm.models.vision_transformer import Block  # 导入Block类


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from torch.optim import AdamW

from types import SimpleNamespace

import matplotlib.cm as cm
from matplotlib import colormaps
import math

import timm
import csv
import json

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


class AttentionMaskDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.samples = []
        self.index_counter = 0
        self.valid_samples = []
        self.blank_sample = self._create_blank_sample()
        
        class_names = sorted(os.listdir(root_dir))
        if not class_names:
            raise ValueError(f"在 {root_dir} 中未找到任何类别目录")
        
        for class_id, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            image_names = sorted(os.listdir(class_path))
            for image_name in image_names:
                image_path = os.path.join(class_path, image_name, 'original.png')
                if not os.path.isfile(image_path):
                    print(f"警告: 图像文件 {image_path} 不存在，跳过")
                    continue
                
                mask_paths = []
                valid = True
                for mask_id in range(16):
                    mask_path = os.path.join(class_path, image_name, f'head_{mask_id}_mask.png')
                    if not os.path.isfile(mask_path):
                        print(f"警告: 掩码文件 {mask_path} 不存在，跳过样本")
                        valid = False
                        break
                    mask_paths.append(mask_path)
                
                if valid:
                    self.samples.append({
                        'image_path': image_path,
                        'mask_paths': mask_paths,
                        'class_id': class_id,
                        'labels': class_id,
                        'class_name': class_name,
                        'image_name': image_name,
                        'index': self.index_counter
                    })
                    self.valid_samples.append(self.index_counter)
                    self.index_counter += 1
        
        print(f"数据集初始化完成，共有 {len(self.valid_samples)}/{len(self.samples)} 个有效样本。")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        for attempt in range(3):
            try:
                sample = self.samples[idx]
                
                try:
                    with Image.open(sample['image_path']) as img:
                        image = img.copy().convert('RGB')
                except (OSError, IOError, Image.DecompressionBombError) as e:
                    print(f"警告: 无法读取图像 {sample['image_path']} (尝试 {attempt+1}/3): {str(e)}")
                    idx = random.choice(self.valid_samples)
                    continue
                
                masks = []
                mask_loaded = True
                for mask_path in sample['mask_paths']:
                    try:
                        with Image.open(mask_path) as mask_img:
                            mask = mask_img.copy().convert('L')
                            masks.append(mask)
                    except (OSError, IOError, Image.DecompressionBombError) as e:
                        print(f"警告: 无法读取掩码 {mask_path} (尝试 {attempt+1}/3): {str(e)}")
                        mask_loaded = False
                        break
                
                if not mask_loaded:
                    idx = random.choice(self.valid_samples)
                    continue
                
                try:
                    if self.image_transform:
                        image = self.image_transform(image)
                    else:
                        image = transforms.ToTensor()(image)
                    
                    transformed_masks = []
                    for mask in masks:
                        if self.mask_transform:
                            mask = self.mask_transform(mask)
                        else:
                            mask = transforms.ToTensor()(mask)
                        transformed_masks.append(mask)
                    
                    masks_tensor = torch.stack(transformed_masks)
                except Exception as e:
                    print(f"警告: 变换处理时出错 (尝试 {attempt+1}/3): {str(e)}")
                    idx = random.choice(self.valid_samples)
                    continue
                
                return {
                    'image': image,
                    'masks': masks_tensor,
                    'class_id': sample['class_id'],
                    'labels': sample['class_id'],
                    'class_name': sample['class_name'],
                    'image_name': sample['image_name'],
                    'index': sample['index']
                }
            
            except Exception as e:
                print(f"严重警告: 加载样本 {idx} 时发生意外错误 (尝试 {attempt+1}/3): {str(e)}")
                if attempt < 2:
                    idx = random.choice(self.valid_samples)
                    continue
                else:
                    print(f"错误: 无法加载样本 {idx}，使用空白样本替代")
                    return self.blank_sample
        
        print(f"错误: 无法加载样本 {idx} (尝试3次失败)，使用空白样本")
        return self.blank_sample
    
    def _create_blank_sample(self, size=(224, 224)):
        blank_image = torch.zeros(3, *size)
        blank_masks = torch.zeros(16, 1, *size)
        return {
            'image': blank_image,
            'masks': blank_masks,
            'class_id': 0,
            'labels': 0,
            'class_name': "blank",
            'image_name': "blank",
            'index': -1  # 特殊值标识问题样本
        }
    
    def validate_sample(self, idx):
        try:
            self.__getitem__(idx)
            return True
        except:
            return False
        

class DeiTMAE(nn.Module):
    def __init__(self, model_name='deit_tiny_patch16_224', pretrained=False, 
                 decoder_dim=512, decoder_depth=4, decoder_num_heads=8, num_classes=1000, local_checkpoint=None):
        super(DeiTMAE, self).__init__()
        
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0, 
            global_pool=''
        )
        
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.embed_dim = self.encoder.embed_dim
        
        self.decoder_dim = decoder_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, decoder_dim),
            *[Block(
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            ) for _ in range(decoder_depth)],
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, 3 * self.patch_size**2),
            nn.Sigmoid()
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
        #for param in self.encoder.parameters():
        #    param.requires_grad = False
        
        print(f"初始化DeiTMAE模型: {model_name}")
        print(f"编码器维度: {self.embed_dim}, 解码器维度: {decoder_dim}")
        print(f"解码器深度: {decoder_depth}, 注意力头数: {decoder_num_heads}")
        
        if local_checkpoint:
            self.load_checkpoint(local_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        print(f"加载本地模型权重：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model_state_dict = self.encoder.state_dict()
        checkpoint_model = checkpoint['model']
        
        filtered_state_dict = {k: v for k, v in checkpoint_model.items() if k in model_state_dict}
        
        self.encoder.load_state_dict(filtered_state_dict, strict=False)
        print(f"成功加载编码器部分的权重：{checkpoint_path}")

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"输入尺寸({H},{W})必须能被patch大小({patch_size})整除"
        
        masked_x = x
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            assert mask.shape[0] == x.shape[0], f"批次大小不匹配: {mask.shape[0]} != {x.shape[0]}"
            
            mask = mask.expand(-1, C, -1, -1)
            
            masked_x = x * mask

        x_patches = self.encoder.patch_embed(masked_x)  # (B, num_patches, embed_dim)
        
        if self.encoder.pos_embed is not None:
            x_patches = x_patches + self.encoder.pos_embed[:, 1:, :]
        
        x_patches = self.encoder.blocks(x_patches)
        x_patches = self.encoder.norm(x_patches)
        
        reconstructions = self.decoder(x_patches)  # (B, num_patches, 3 * patch_size**2)
        
        patch_dim = int(patch_size**2 * 3)
        reconstructions = reconstructions.reshape(B, -1, patch_dim)  # (B, num_patches, patch_dim)
        
        h, w = H // patch_size, W // patch_size
        reconstructions = reconstructions.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size, h, w)
        reconstructions = reconstructions.permute(0, 1, 4, 2, 5, 3).reshape(B, 3, H, W)

        x_class = self.encoder.patch_embed(x)
        if self.encoder.pos_embed is not None:
            x_class = x_class + self.encoder.pos_embed[:, 1:, :]
        
        x_class = self.encoder.blocks(x_class)
        x_class = self.encoder.norm(x_class)
        
        class_token = torch.mean(x_class, dim=1)  # (B, embed_dim)
        class_output = self.classifier(class_token)
        
        return reconstructions, class_output
    

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f'top{k}'] = correct_k.mul_(1.0 / batch_size).item()
        return res


class MAEBoostingTrainer:
    def __init__(self, train_dataset, val_dataset, device, num_learners=16, learning_rate=0.001, batch_size=32,
                 model_name='deit_base_patch16_224', num_classes=1000, saved_models_dir="./saved_models_ensemble",
                 logs_dir="./training_logs_ensemble", **model_kwargs):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        
        self.trained_learners = []
        
        self.precomputed_summed_logits = None
        self.train_mask_orders = {}
        self.val_mask_orders = {}
        
        self.saved_models_dir = saved_models_dir
        self.logs_dir = logs_dir
        os.makedirs(self.saved_models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.training_logs = {
            "epoch_wise_ensemble_val_acc_top1": [],
            "epoch_wise_ensemble_val_acc_top5": [],
            "learner_wise_losses_recon": [],
            "learner_wise_losses_cls_ensemble": [],
        }
    
    def calculate_mask_response(self, mask):
        return torch.mean(mask)
    
    def get_sorted_mask_indices(self, masks):
        responses = torch.mean(masks, dim=[1, 2, 3])
        sorted_indices = torch.argsort(responses, descending=True).tolist()
        return sorted_indices
    
    def get_mask_for_index(self, masks, idx, dataset_type='train', k=1):
        mask_orders = self.train_mask_orders if dataset_type == 'train' else self.val_mask_orders
        
        if idx not in mask_orders:
            mask_orders[idx] = self.get_sorted_mask_indices(masks)
        
        mask_indices = mask_orders[idx][:k]
        
        selected_masks = masks[mask_indices]  # (k, 1, H, W)
        if selected_masks.dim() == 4:
            selected_masks = selected_masks.squeeze(1)  # (k, H, W)
        combined_mask = torch.max(selected_masks, dim=0)[0]  # (H, W)
        
        return combined_mask
    
    def _precompute_summed_logits(self):
        if not self.trained_learners:
            self.precomputed_summed_logits = None
            return

        print(f"\n--- 预计算前 {len(self.trained_learners)} 个学习器集成的 Logits 之和 ---")
        N = len(self.train_dataset)
        self.precomputed_summed_logits = torch.zeros(N, self.num_classes, device='cpu')

        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        with torch.no_grad():
            for learner_idx, model in enumerate(self.trained_learners):
                model.to(self.device)
                model.eval()
                print(f"  > 正在处理学习器 {learner_idx + 1}/{len(self.trained_learners)}")
                pbar = tqdm(loader, desc=f"预计算 Logits (Learner {learner_idx+1})", ncols=100)
                for batch in pbar:
                    images = batch["image"].to(self.device)
                    masks = batch["masks"].to(self.device)
                    idxs = batch["index"]

                    batch_masks_to_use = torch.stack([
                        self.get_mask_for_index(masks[i], idxs[i].item(), 'train', learner_idx + 1)
                        for i in range(len(idxs))
                    ]).unsqueeze(1).to(self.device)
                    
                    _, logits = model(images, mask=batch_masks_to_use)
                    self.precomputed_summed_logits[idxs] += logits.cpu()
                
                model.to('cpu') # 释放显存
        
        torch.cuda.empty_cache()

    def train_learner(self, learner_idx, num_epochs=10):
        current_model = DeiTMAE(
            model_name=self.model_name, num_classes=self.num_classes, **self.model_kwargs
        ).to(self.device)
        optimizer = optim.AdamW(current_model.parameters(), lr=self.learning_rate)
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        criterion_recon = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        
        k = learner_idx + 1

        for epoch in range(num_epochs):
            current_model.train()
            total_loss_recon = 0.0
            total_loss_cls_ensemble = 0.0
            
            pbar = tqdm(
                train_loader,
                desc=f"Learner {k}/{self.num_learners} Epoch {epoch+1}/{num_epochs} [训练]",
                ncols=120
            )

            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["masks"].to(self.device)
                labels = batch["labels"].to(self.device)
                idxs = batch["index"]

                batch_masks_for_current = torch.stack([
                    self.get_mask_for_index(masks[i], idxs[i].item(), 'train', k)
                    for i in range(len(idxs))
                ]).unsqueeze(1).to(self.device)

                recon, current_logits = current_model(images, mask=batch_masks_for_current)
                
                loss_recon = criterion_recon(recon, images)

                if self.precomputed_summed_logits is not None:
                    prev_summed_logits = self.precomputed_summed_logits[idxs].to(self.device)
                    ensemble_logits = (prev_summed_logits + current_logits) / k
                else:
                    ensemble_logits = current_logits
                
                loss_cls_ensemble = criterion_cls(ensemble_logits, labels)
                
                total_loss = loss_recon + loss_cls_ensemble
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss_recon += loss_recon.item()
                total_loss_cls_ensemble += loss_cls_ensemble.item()

                pbar.set_postfix({
                    "L_recon": f"{total_loss_recon / (pbar.n + 1):.4f}",
                    "L_cls_ensemble": f"{total_loss_cls_ensemble / (pbar.n + 1):.4f}"
                })
            
            current_model.eval()


            temp_ensemble_for_validation = []
            for m in self.trained_learners:
                temp_ensemble_for_validation.append(m.to(self.device))
            temp_ensemble_for_validation.append(current_model)
            val_acc1, val_acc5 = self.validate_ensemble(temp_ensemble_for_validation, epoch, num_epochs)

            for m in temp_ensemble_for_validation[:-1]:
                m.to('cpu')

            current_model.train()
            
            self.training_logs["epoch_wise_ensemble_val_acc_top1"].append(val_acc1)
            self.training_logs["epoch_wise_ensemble_val_acc_top5"].append(val_acc5)

        current_model.eval()
        current_model.to('cpu')
        torch.cuda.empty_cache()
        self.trained_learners.append(current_model)

        model_path = os.path.join(self.saved_models_dir, f"learner_{learner_idx}.pth")
        torch.save(current_model.state_dict(), model_path)
        print(f"保存学习器 {k} 至: {model_path}")
        
        self.training_logs["learner_wise_losses_recon"].append(total_loss_recon / len(train_loader))
        self.training_logs["learner_wise_losses_cls_ensemble"].append(total_loss_cls_ensemble / len(train_loader))

    def validate_ensemble(self, ensemble_models, epoch, num_epochs):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        num_models_in_ensemble = len(ensemble_models)
        
        total_val_acc_top1 = 0.0
        total_val_acc_top5 = 0.0

        pbar = tqdm(
            val_loader,
            desc=f"Ensemble {num_models_in_ensemble} Epoch {epoch+1}/{num_epochs} [验证]",
            ncols=120
        )

        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["masks"].to(self.device)
                labels = batch["labels"].to(self.device)
                idxs = batch["index"]

                batch_ensemble_logits = torch.zeros(images.size(0), self.num_classes).to(self.device)
                
        
                for model_idx, model in enumerate(ensemble_models):
                    
                    batch_masks_to_use = torch.stack([
                        self.get_mask_for_index(masks[i], idxs[i].item(), 'val', model_idx + 1)
                        for i in range(len(idxs))
                    ]).unsqueeze(1).to(self.device)

                    _, logits = model(images, mask=batch_masks_to_use)
                    batch_ensemble_logits += logits
                
                final_ensemble_logits = batch_ensemble_logits / num_models_in_ensemble

                acc = accuracy(final_ensemble_logits, labels, topk=(1, 5))
                total_val_acc_top1 += acc['top1']
                total_val_acc_top5 += acc['top5']

                pbar.set_postfix({
                    "Acc@1": f"{total_val_acc_top1 / (pbar.n + 1) * 100:.2f}%",
                    "Acc@5": f"{total_val_acc_top5 / (pbar.n + 1) * 100:.2f}%"
                })

        val_acc1 = total_val_acc_top1 / len(val_loader)
        val_acc5 = total_val_acc_top5 / len(val_loader)
        print(f"\n[验证] 集成模型 (共 {num_models_in_ensemble} 个) - Top-1 Acc: {val_acc1*100:.2f}%, Top-5 Acc: {val_acc5*100:.2f}%")
        return val_acc1, val_acc5

    ### MODIFIED ###
    def train_all_learners(self, num_epochs=10):
        for i in range(self.num_learners):
            print(f"\n{'='*20} 开始训练学习器 {i + 1}/{self.num_learners} {'='*20}")
            
            self._precompute_summed_logits()
            
            self.train_learner(i, num_epochs)

        self.save_training_logs()
        print("\n所有学习器训练完毕！")
    
    def precompute_val_mask_orders(self):
        print("\n预计算验证集掩码排序...")
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        for batch in tqdm(val_loader, desc="预计算验证集掩码"):
            masks = batch["masks"]
            idxs = batch["index"]
            
            for i, sample_idx in enumerate(idxs):
                idx_val = sample_idx.item()
                if idx_val not in self.val_mask_orders:
                    self.val_mask_orders[idx_val] = self.get_sorted_mask_indices(masks[i])
    
    def save_training_logs(self, final_val_acc_top1):
        log_file = os.path.join(self.logs_dir, "training_logs.json")
        
        logs = {
            "learners": self.num_learners,
            "epochs_per_learner": len(self.training_logs["train_acc_top1"][0]) if self.training_logs["train_acc_top1"] else 0,
            "final_validation_accuracy_top1": [acc * 100 for acc in final_val_acc_top1],
            "training_history": {
                "reconstruction_loss": self.training_logs["losses_recon"],
                "classification_loss": self.training_logs["losses_cls"],
                "training_accuracy_top1": [[acc * 100 for acc in accs] for accs in self.training_logs["train_acc_top1"]],
                "training_accuracy_top5": [[acc * 100 for acc in accs] for accs in self.training_logs["train_acc_top5"]],
                "validation_accuracy_top1": [[acc * 100 for acc in accs] for accs in self.training_logs["val_acc_top1"]],
                "validation_accuracy_top5": [[acc * 100 for acc in accs] for accs in self.training_logs["val_acc_top5"]]
            },
            "best_models": self.training_logs["best_models"]
        }
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        
        print(f"训练日志保存至: {log_file}")
        
        csv_file = os.path.join(self.logs_dir, "training_metrics.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            headers = ["epoch"]
            for i in range(self.num_learners):
                headers += [
                    f"learner_{i+1}_loss_recon", 
                    f"learner_{i+1}_loss_cls",
                    f"learner_{i+1}_train_acc_top1",
                    f"learner_{i+1}_train_acc_top5",
                    f"learner_{i+1}_val_acc_top1",
                    f"learner_{i+1}_val_acc_top5"
                ]
            
            writer.writerow(headers)
            
            num_epochs = len(logs["training_history"]["training_accuracy_top1"][0])
            for epoch in range(num_epochs):
                row = [epoch + 1]
                for learner in range(self.num_learners):
                    try:
                        row += [
                            logs["training_history"]["reconstruction_loss"][learner][epoch],
                            logs["training_history"]["classification_loss"][learner][epoch],
                            logs["training_history"]["training_accuracy_top1"][learner][epoch],
                            logs["training_history"]["training_accuracy_top5"][learner][epoch],
                            logs["training_history"]["validation_accuracy_top1"][learner][epoch],
                            logs["training_history"]["validation_accuracy_top5"][learner][epoch]
                        ]
                    except IndexError:
                        row += ["", "", "", "", "", ""] # 不足epoch填充空值
                
                writer.writerow(row)
        
        print(f"训练指标保存至: {csv_file}")
            
class Args:
    def __init__(self):
        self.arch = 'vit_large'
        self.patch_size = 16
        self.pretrained_weights = './saved/Vit-L/Block/checkpoint.pth'
        self.checkpoint_key = 'teacher'
        self.image_path = '/media/data/Imagenet/val' #'/media/data/Imagenet/train/n01440764/n01440764_10026.JPEG'
        self.data_path = '/media/data/Imagenet/val'
        self.batch_size = 32
        self.image_size = [224, 224]  # 原始图像尺寸
        self.output_dir = './output'
        self.show_pics = 100
        self.threshold = 0.6

args = Args()

image_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

mask_transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.ToTensor()
])

dataset_train = AttentionMaskDataset(
        root_dir= '/media/data/Imagenet/Mask/train', # './output/val', #
        image_transform=image_transform,
        mask_transform=mask_transform
    )

dataset_val = AttentionMaskDataset(
        root_dir='./output/val',
        image_transform=image_transform,
        mask_transform=mask_transform
    )

trainer = MAEBoostingTrainer(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    device=DEVICE,
    num_learners=16,
    learning_rate=1e-4,
    batch_size=64,
    model_name='deit_small_patch16_224',
    pretrained=True,
    decoder_dim=512,
    decoder_depth=4,
    decoder_num_heads=8
)


trainer.train_all_learners(num_epochs=10)
