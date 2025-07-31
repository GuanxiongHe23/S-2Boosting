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
        """
        改进的数据集类，支持独立的图像和掩码变换，包含健壮的异常处理
        """
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.samples = []
        self.index_counter = 0
        self.valid_samples = []  # 跟踪有效样本
        self.blank_sample = self._create_blank_sample()  # 创建空白样本作为备份
        
        # 收集样本
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
                
                # 检查所有16个掩码
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
        # 最多尝试3次加载样本
        for attempt in range(3):
            try:
                sample = self.samples[idx]
                
                # 尝试打开图像
                try:
                    with Image.open(sample['image_path']) as img:
                        image = img.copy().convert('RGB')
                except (OSError, IOError, Image.DecompressionBombError) as e:
                    print(f"警告: 无法读取图像 {sample['image_path']} (尝试 {attempt+1}/3): {str(e)}")
                    idx = random.choice(self.valid_samples)  # 尝试另一个有效样本
                    continue
                
                # 尝试加载所有掩码
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
                    idx = random.choice(self.valid_samples)  # 尝试另一个有效样本
                    continue
                
                # 应用变换
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
                    idx = random.choice(self.valid_samples)  # 尝试另一个有效样本
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
                if attempt < 2:  # 如果不是最后一次尝试
                    idx = random.choice(self.valid_samples)  # 尝试另一个有效样本
                    continue
                else:
                    print(f"错误: 无法加载样本 {idx}，使用空白样本替代")
                    return self.blank_sample
        
        # 如果三次尝试都失败，返回空白样本
        print(f"错误: 无法加载样本 {idx} (尝试3次失败)，使用空白样本")
        return self.blank_sample
    
    def _create_blank_sample(self, size=(224, 224)):
        """创建一个空白样本作为故障恢复选项"""
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
        """验证特定样本是否有效且可加载"""
        try:
            self.__getitem__(idx)
            return True
        except:
            return False
        

class DeiTMAE(nn.Module):
    def __init__(self, model_name='deit_tiny_patch16_224', pretrained=False, 
                 decoder_dim=512, decoder_depth=4, decoder_num_heads=8, num_classes=1000, local_checkpoint=None):
        """
        基于DeiT的MAE模型，使用预训练的DeiT作为编码器，并加入分类头和重构头
        
        参数:
            model_name (str): DeiT模型名称
            pretrained (bool): 是否加载预训练权重
            decoder_dim (int): 解码器维度
            decoder_depth (int): 解码器层数
            decoder_num_heads (int): 解码器注意力头数
            num_classes (int): 分类任务的类别数
            local_checkpoint (str): 本地预训练模型路径 (默认 None)
        """
        super(DeiTMAE, self).__init__()
        
        # 加载预训练的DeiT模型作为编码器
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,  # 设置为False，因为我们将手动加载权重
            num_classes=0,  # 移除分类头
            global_pool=''   # 返回所有token
        )
        
        # 获取模型配置
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.embed_dim = self.encoder.embed_dim
        
        # 解码器配置
        self.decoder_dim = decoder_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, decoder_dim),
            *[Block(  # 使用timm中的Block
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            ) for _ in range(decoder_depth)],
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, 3 * self.patch_size**2),  # 输出RGB像素值
            nn.Sigmoid()
        )
        
        # 可学习的掩码token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 分类头
        self.classifier = nn.Linear(self.embed_dim, num_classes)  # 分类头
        
        # 冻结编码器参数（可选）
        #for param in self.encoder.parameters():
        #    param.requires_grad = False
        
        print(f"初始化DeiTMAE模型: {model_name}")
        print(f"编码器维度: {self.embed_dim}, 解码器维度: {decoder_dim}")
        print(f"解码器深度: {decoder_depth}, 注意力头数: {decoder_num_heads}")
        
        # 如果提供了本地检查点路径，则加载模型权重
        if local_checkpoint:
            self.load_checkpoint(local_checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        """
        从本地加载模型检查点权重
        
        参数:
            checkpoint_path (str): 本地检查点文件路径
        """
        print(f"加载本地模型权重：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 加载编码器部分的权重
        model_state_dict = self.encoder.state_dict()
        checkpoint_model = checkpoint['model']
        
        # 筛选出与编码器匹配的权重
        filtered_state_dict = {k: v for k, v in checkpoint_model.items() if k in model_state_dict}
        
        # 加载匹配的权重
        self.encoder.load_state_dict(filtered_state_dict, strict=False)
        print(f"成功加载编码器部分的权重：{checkpoint_path}")

    def forward(self, x, mask=None):
        """
        修改后的前向传播：
        - 分类输出始终在原始图像上计算
        - 重构输出使用掩码后的图像
        
        参数:
            x (Tensor): 输入图像 (B, C, H, W)
            mask (Tensor): 来自数据集的掩码 (B, 1, H, W)
        
        返回:
            Tensor: 重建图像 (B, C, H, W)
            Tensor: 分类logits (B, num_classes)
        """
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        # 确保输入尺寸兼容
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"输入尺寸({H},{W})必须能被patch大小({patch_size})整除"
        
        # 1. 应用掩码（如果提供）
        masked_x = x
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 将掩码扩展为 (B, 1, H, W)
            assert mask.shape[0] == x.shape[0], f"批次大小不匹配: {mask.shape[0]} != {x.shape[0]}"
            
            # 将掩码扩展为 (B, C, H, W) 以便与图像相乘
            mask = mask.expand(-1, C, -1, -1)  # 扩展掩码形状
            
            # 应用掩码
            masked_x = x * mask  # 使用数据集提供的掩码来遮盖图像
        
        # ========== 重构分支：使用掩码图像 ==========
        # 2. 提取图像块并编码
        x_patches = self.encoder.patch_embed(masked_x)  # (B, num_patches, embed_dim)
        
        # 添加位置编码
        if self.encoder.pos_embed is not None:
            # DeiT使用CLS token，所以跳过第一个位置编码
            x_patches = x_patches + self.encoder.pos_embed[:, 1:, :]
        
        # 3. 应用DeiT编码器
        x_patches = self.encoder.blocks(x_patches)
        x_patches = self.encoder.norm(x_patches)
        
        # 4. 解码重建
        reconstructions = self.decoder(x_patches)  # (B, num_patches, 3 * patch_size**2)
        
        # 5. 重塑为图像格式
        patch_dim = int(patch_size**2 * 3)
        reconstructions = reconstructions.reshape(B, -1, patch_dim)  # (B, num_patches, patch_dim)
        
        # 重塑为(B, C, H, W)
        h, w = H // patch_size, W // patch_size
        reconstructions = reconstructions.permute(0, 2, 1).reshape(B, 3, patch_size, patch_size, h, w)
        reconstructions = reconstructions.permute(0, 1, 4, 2, 5, 3).reshape(B, 3, H, W)
        
        # ========== 分类分支：使用原始图像 ==========
        # 为分类分支单独处理原始图像
        x_class = self.encoder.patch_embed(x)  # 使用原始图像
        if self.encoder.pos_embed is not None:
            x_class = x_class + self.encoder.pos_embed[:, 1:, :]
        
        # 应用DeiT编码器
        x_class = self.encoder.blocks(x_class)
        x_class = self.encoder.norm(x_class)
        
        # 分类输出：使用所有token的均值作为表示
        # (在原始实现中，DeiT使用CLS token，但这里使用全局平均池化)
        class_token = torch.mean(x_class, dim=1)  # (B, embed_dim)
        class_output = self.classifier(class_token)
        
        return reconstructions, class_output
    

# 新增: 一个标准的准确率计算辅助函数
def accuracy(output, target, topk=(1,)):
    """根据指定的k值，计算top-k预测的准确率"""
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
        """
        新的集成训练器，直接优化集成模型的损失。
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        
        # 存储训练好的、冻结的学习器
        self.trained_learners = []
        
        # 预计算的logits和掩码排序
        self.precomputed_summed_logits = None
        self.train_mask_orders = {}
        self.val_mask_orders = {}
        
        # 路径设置
        self.saved_models_dir = saved_models_dir
        self.logs_dir = logs_dir
        os.makedirs(self.saved_models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # 训练日志
        self.training_logs = {
            "epoch_wise_ensemble_val_acc_top1": [],
            "epoch_wise_ensemble_val_acc_top5": [],
            "learner_wise_losses_recon": [],
            "learner_wise_losses_cls_ensemble": [],
        }
    
    def calculate_mask_response(self, mask):
        """计算掩码的平均响应强度"""
        return torch.mean(mask)
    
    def get_sorted_mask_indices(self, masks):
        """计算并排序掩码响应强度"""
        responses = torch.mean(masks, dim=[1, 2, 3])
        sorted_indices = torch.argsort(responses, descending=True).tolist()
        return sorted_indices
    
    def get_mask_for_index(self, masks, idx, dataset_type='train', k=1):
        """获取指定样本和数据集类型的前k个掩码的并集"""
        # 确定使用哪个掩码排序缓存
        mask_orders = self.train_mask_orders if dataset_type == 'train' else self.val_mask_orders
        
        # 确保样本索引存在于排序缓存中
        if idx not in mask_orders:
            mask_orders[idx] = self.get_sorted_mask_indices(masks)
        
        # 获取前k个mask的索引
        mask_indices = mask_orders[idx][:k]
        
        # 计算前k个mask的并集
        selected_masks = masks[mask_indices]  # (k, 1, H, W)
        if selected_masks.dim() == 4:
            selected_masks = selected_masks.squeeze(1)  # (k, H, W)
        combined_mask = torch.max(selected_masks, dim=0)[0]  # (H, W)
        
        return combined_mask
    
    ### NEW ###
    def _precompute_summed_logits(self):
        """
        预计算当前集成模型（不含新学习器）在整个训练集上的logits之和。
        """
        if not self.trained_learners:
            self.precomputed_summed_logits = None
            return

        print(f"\n--- 预计算前 {len(self.trained_learners)} 个学习器集成的 Logits 之和 ---")
        N = len(self.train_dataset)
        self.precomputed_summed_logits = torch.zeros(N, self.num_classes, device='cpu')

        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        with torch.no_grad():
            # 遍历每个已训练好的模型
            for learner_idx, model in enumerate(self.trained_learners):
                model.to(self.device)
                model.eval()
                print(f"  > 正在处理学习器 {learner_idx + 1}/{len(self.trained_learners)}")
                pbar = tqdm(loader, desc=f"预计算 Logits (Learner {learner_idx+1})", ncols=100)
                for batch in pbar:
                    images = batch["image"].to(self.device)
                    masks = batch["masks"].to(self.device)
                    idxs = batch["index"]

                    # 为当前模型获取正确的掩码 (k = learner_idx + 1)
                    batch_masks_to_use = torch.stack([
                        self.get_mask_for_index(masks[i], idxs[i].item(), 'train', learner_idx + 1)
                        for i in range(len(idxs))
                    ]).unsqueeze(1).to(self.device)
                    
                    _, logits = model(images, mask=batch_masks_to_use)
                    self.precomputed_summed_logits[idxs] += logits.cpu()
                
                model.to('cpu') # 释放显存
        
        torch.cuda.empty_cache()

    ### MODIFIED ###
    def train_learner(self, learner_idx, num_epochs=10):
        """
        训练单个学习器，使其加入集成后能最小化总的集成损失。
        learner_idx 从 0 开始。
        """
        # 创建新模型和优化器
        current_model = DeiTMAE(
            model_name=self.model_name, num_classes=self.num_classes, **self.model_kwargs
        ).to(self.device)
        optimizer = optim.AdamW(current_model.parameters(), lr=self.learning_rate)
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        criterion_recon = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        
        # k 是当前集成中的模型总数
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

                # 获取当前模型 (k-th learner) 的掩码
                batch_masks_for_current = torch.stack([
                    self.get_mask_for_index(masks[i], idxs[i].item(), 'train', k)
                    for i in range(len(idxs))
                ]).unsqueeze(1).to(self.device)

                # 1. 前向传播当前模型
                recon, current_logits = current_model(images, mask=batch_masks_for_current)
                
                # 2. 计算重构损失 (只针对当前模型)
                loss_recon = criterion_recon(recon, images)

                # 3. 计算集成分类损失
                if self.precomputed_summed_logits is not None:
                    # 获取预计算的前 k-1 个模型的 logits 之和
                    prev_summed_logits = self.precomputed_summed_logits[idxs].to(self.device)
                    # 计算新的集成 logits (平均)
                    ensemble_logits = (prev_summed_logits + current_logits) / k
                else:
                    # 第一个学习器，集成 logits 就是它自己
                    ensemble_logits = current_logits
                
                loss_cls_ensemble = criterion_cls(ensemble_logits, labels)
                
                # 4. 总损失与反向传播
                total_loss = loss_recon + loss_cls_ensemble
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # 统计
                total_loss_recon += loss_recon.item()
                total_loss_cls_ensemble += loss_cls_ensemble.item()

                pbar.set_postfix({
                    "L_recon": f"{total_loss_recon / (pbar.n + 1):.4f}",
                    "L_cls_ensemble": f"{total_loss_cls_ensemble / (pbar.n + 1):.4f}"
                })
            
            # --- 验证阶段 ---
            current_model.eval() # 切换到评估模式

            # 准备用于验证的集成列表 (所有模型都在CPU上，除了当前模型)
            # 我们需要确保所有模型都在GPU上进行验证
            temp_ensemble_for_validation = []
            for m in self.trained_learners:
                temp_ensemble_for_validation.append(m.to(self.device))
            temp_ensemble_for_validation.append(current_model) # current_model 已经在 device 上

            val_acc1, val_acc5 = self.validate_ensemble(temp_ensemble_for_validation, epoch, num_epochs)

            # 验证结束后，将临时的模型移回CPU
            for m in temp_ensemble_for_validation[:-1]: # 不包括 current_model
                m.to('cpu')

            # 将当前模型切换回训练模式，以进行下一个epoch
            current_model.train() # 切换回训练模式

            # 记录日志
            self.training_logs["epoch_wise_ensemble_val_acc_top1"].append(val_acc1)
            self.training_logs["epoch_wise_ensemble_val_acc_top5"].append(val_acc5)

        # 训练结束后，冻结当前模型并加入到集成列表中
        current_model.eval()
        current_model.to('cpu')
        torch.cuda.empty_cache()
        self.trained_learners.append(current_model)

        # 保存模型
        model_path = os.path.join(self.saved_models_dir, f"learner_{learner_idx}.pth")
        torch.save(current_model.state_dict(), model_path)
        print(f"保存学习器 {k} 至: {model_path}")
        
        # 记录该学习器训练结束后的损失
        self.training_logs["learner_wise_losses_recon"].append(total_loss_recon / len(train_loader))
        self.training_logs["learner_wise_losses_cls_ensemble"].append(total_loss_cls_ensemble / len(train_loader))

    ### MODIFIED ###
    def validate_ensemble(self, ensemble_models, epoch, num_epochs):
        """在验证集上评估当前的完整集成模型"""
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
                
                # 遍历集成中的每个模型
                for model_idx, model in enumerate(ensemble_models):
                    
                    # 获取该模型对应的掩码 (k = model_idx + 1)
                    batch_masks_to_use = torch.stack([
                        self.get_mask_for_index(masks[i], idxs[i].item(), 'val', model_idx + 1)
                        for i in range(len(idxs))
                    ]).unsqueeze(1).to(self.device)

                    _, logits = model(images, mask=batch_masks_to_use)
                    batch_ensemble_logits += logits
                
                # 计算平均 logits
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
        """按顺序训练所有学习器"""
        for i in range(self.num_learners):
            print(f"\n{'='*20} 开始训练学习器 {i + 1}/{self.num_learners} {'='*20}")
            
            # 1. 预计算前 i 个模型的 logits 之和 (如果 i>0)
            self._precompute_summed_logits()
            
            # 2. 训练第 i 个学习器 (总共 i+1 个模型在集成中)
            self.train_learner(i, num_epochs)

        self.save_training_logs()
        print("\n所有学习器训练完毕！")
    
    def precompute_val_mask_orders(self):
        """预计算验证集的掩码排序"""
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
        """MODIFIED: 保存训练日志，包含Top-5准确率"""
        log_file = os.path.join(self.logs_dir, "training_logs.json")
        
        # 创建日志字典
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
        
        # 保存为JSON
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        
        print(f"训练日志保存至: {log_file}")
        
        # 保存为CSV以便分析
        csv_file = os.path.join(self.logs_dir, "training_metrics.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # 写入标题
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
            
            # 写入数据
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

# 准备数据集
image_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

# 掩码变换 - 只需要调整大小和转换为张量
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
