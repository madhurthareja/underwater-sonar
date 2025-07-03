#!/usr/bin/env python3
"""
YOLOv11 Model Architecture Customization
For advanced training modifications
"""

import torch
import torch.nn as nn
import yaml
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomYOLOv11(nn.Module):
    """Custom YOLOv11 architecture for sonar image detection"""
    
    def __init__(self, cfg_path=None, nc=10, anchors=None):
        super().__init__()
        self.nc = nc  # number of classes
        self.model = None
        
        if cfg_path:
            self.load_custom_config(cfg_path)
        else:
            self.create_default_config()
    
    def load_custom_config(self, cfg_path):
        """Load custom model configuration"""
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        logger.info(f"Loaded custom config: {cfg_path}")
    
    def create_default_config(self):
        """Create default configuration optimized for sonar images"""
        self.cfg = {
            # Model architecture
            'backbone': [
                [-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
                [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
                [-1, 3, C2f, [128, True]],
                [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
                [-1, 6, C2f, [256, True]],
                [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
                [-1, 6, C2f, [512, True]],
                [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
                [-1, 3, C2f, [1024, True]],
                [-1, 1, SPPF, [1024, 5]],  # 9
            ],
            
            # Neck
            'neck': [
                [-1, 1, nn.Upsample, [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                [-1, 3, C2f, [512]],  # 12
                
                [-1, 1, nn.Upsample, [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                [-1, 3, C2f, [256]],  # 15 (P3/8-small)
                
                [-1, 1, Conv, [256, 3, 2]],
                [[-1, 12], 1, 'Concat', [1]],  # cat head P4
                [-1, 3, C2f, [512]],  # 18 (P4/16-medium)
                
                [-1, 1, Conv, [512, 3, 2]],
                [[-1, 9], 1, 'Concat', [1]],  # cat head P5
                [-1, 3, C2f, [1024]],  # 21 (P5/32-large)
            ],
            
            # Head
            'head': [
                [[15, 18, 21], 1, Detect, [self.nc]],  # Detect(P3, P4, P5)
            ]
        }
    
    def modify_for_sonar_images(self):
        """Modify architecture specifically for sonar images"""
        # Add custom preprocessing layers for sonar-specific features
        self.sonar_preprocessor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),  # Larger receptive field
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Add attention mechanism for sonar features
        self.attention = SonarAttention(64)
        
        logger.info("Modified architecture for sonar images")
    
    def create_model_yaml(self, save_path):
        """Create YAML configuration file for custom model"""
        config = {
            'nc': self.nc,
            'depth_multiple': 0.33,
            'width_multiple': 0.25,
            'backbone': self.cfg['backbone'],
            'head': self.cfg['head']
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Model configuration saved to: {save_path}")
        return save_path


class SonarAttention(nn.Module):
    """Attention mechanism for sonar image features"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_att = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_att
        
        return x


class SonarLoss(nn.Module):
    """Custom loss function for sonar object detection"""
    
    def __init__(self, nc=10, device='cuda'):
        super().__init__()
        self.nc = nc
        self.device = device
        
        # Standard YOLO losses
        self.box_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        
        # Sonar-specific weights
        self.sonar_weights = {
            'human body': 2.0,  # Higher weight for human detection
            'ROV': 1.5,
            'metal bucket': 1.3,
            'ball': 1.0,
            'cube': 1.0,
            'cylinder': 1.0,
            'model': 1.0,
            'circle cage': 1.0,
            'square cage': 1.0,
            'plane model': 1.0
        }
    
    def forward(self, predictions, targets):
        """Calculate custom loss for sonar detection"""
        # Apply class-specific weights
        weighted_loss = 0
        
        # This is a simplified version - actual implementation would be more complex
        for pred, target in zip(predictions, targets):
            # Apply sonar-specific weighting
            class_id = target['class_id']
            weight = list(self.sonar_weights.values())[class_id] if class_id < len(self.sonar_weights) else 1.0
            
            # Calculate weighted loss (simplified)
            loss = self.box_loss(pred['bbox'], target['bbox']) + self.cls_loss(pred['class'], target['class'])
            weighted_loss += loss * weight
        
        return weighted_loss


def create_custom_training_config():
    """Create custom training configuration for sonar images"""
    config = {
        # Data augmentation optimized for sonar images
        'augmentation': {
            'hsv_h': 0.005,  # Reduced hue variation for sonar
            'hsv_s': 0.3,    # Reduced saturation
            'hsv_v': 0.2,    # Reduced value variation
            'degrees': 5.0,   # Reduced rotation
            'translate': 0.05, # Reduced translation
            'scale': 0.2,     # Reduced scaling
            'shear': 1.0,     # Reduced shear
            'perspective': 0.0, # No perspective for sonar
            'flipud': 0.0,    # No vertical flip
            'fliplr': 0.5,    # Horizontal flip OK
            'mosaic': 0.5,    # Reduced mosaic
            'mixup': 0.0,     # No mixup for sonar
        },
        
        # Training parameters
        'training': {
            'epochs': 200,
            'batch_size': 4,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'patience': 20,
            'save_period': 10,
        },
        
        # Model parameters
        'model': {
            'depth_multiple': 0.33,
            'width_multiple': 0.25,
            'nc': 10,
            'anchors': None,  # Auto-anchor
        },
        
        # Loss weights
        'loss': {
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'focal_loss_gamma': 1.5,
            'label_smoothing': 0.0,
        }
    }
    
    return config


def clone_ultralytics_repo(repo_dir='ultralytics_repo'):
    """Clone the official Ultralytics repository for modifications"""
    import subprocess
    
    if not os.path.exists(repo_dir):
        logger.info("Cloning Ultralytics repository...")
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ultralytics/ultralytics.git', repo_dir], check=True)
            logger.info(f"Repository cloned to {repo_dir}")
            
            # Create symbolic links for easy modification
            if not os.path.exists('ultralytics'):
                os.symlink(f'{repo_dir}/ultralytics', 'ultralytics')
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
    else:
        logger.info(f"Repository already exists at {repo_dir}")
        return True


def create_custom_model_config(save_path='configs/custom_yolo11_sonar.yaml'):
    """Create custom model configuration file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    config = {
        # Model configuration
        'nc': 10,  # number of classes
        'depth_multiple': 0.33,  # model depth multiple
        'width_multiple': 0.25,  # layer channel multiple
        
        # Anchors (will be auto-calculated if None)
        'anchors': None,
        
        # YOLOv11n backbone
        'backbone': [
            # [from, repeats, module, args]
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
            [-1, 2, 'C2f', [128, True]],
            [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
            [-1, 2, 'C2f', [256, True]],
            [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
            [-1, 2, 'C2f', [512, True]],
            [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
            [-1, 2, 'C2f', [1024, True]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
        ],
        
        # YOLOv11n head
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 2, 'C2f', [512]],  # 12
            
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 2, 'C2f', [256]],  # 15 (P3/8-small)
            
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 12], 1, 'Concat', [1]],  # cat head P4
            [-1, 2, 'C2f', [512]],  # 18 (P4/16-medium)
            
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 9], 1, 'Concat', [1]],  # cat head P5
            [-1, 2, 'C2f', [1024]],  # 21 (P5/32-large)
            
            [[15, 18, 21], 1, 'Detect', [10]],  # Detect(P3, P4, P5)
        ]
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Custom model config saved to: {save_path}")
    return save_path


def main():
    """Main function for model customization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv11 Model Customization')
    parser.add_argument('--clone-repo', action='store_true', help='Clone Ultralytics repository')
    parser.add_argument('--create-config', action='store_true', help='Create custom model config')
    parser.add_argument('--config-path', type=str, default='../configs/custom_yolo11_sonar.yaml', help='Config save path')
    
    args = parser.parse_args()
    
    if args.clone_repo:
        clone_ultralytics_repo()
    
    if args.create_config:
        create_custom_model_config(args.config_path)
        
        # Also create training config
        training_config = create_custom_training_config()
        training_config_path = args.config_path.replace('.yaml', '_training.yaml')
        
        with open(training_config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        logger.info(f"Training config saved to: {training_config_path}")
    
    logger.info("Model customization setup completed!")


if __name__ == "__main__":
    main()
