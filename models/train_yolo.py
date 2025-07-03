#!/usr/bin/env python3
"""
YOLOv11 Training Script for UATD Sonar Dataset
"""

import torch
import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(self, config_path=None):
        """Initialize YOLO trainer with configuration"""
        self.config = self.load_config(config_path)
        self.device = self.setup_device()
        self.model = None
        
    def load_config(self, config_path):
        """Load training configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                'model_size': '/home/madhurthareja/sonar/yolo11n.pt',  
                'data_yaml': '/home/madhurthareja/sonar/dataset/data.yaml',
                'epochs': 100,
                'batch_size': 4,  
                'image_size': 640,
                'patience': 15,
                'save_period': 10,
                'workers': 4,
                'optimizer': 'AdamW',
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': False,
                'copy_paste': 0.0,
                'auto_augment': 'randaugment',
                'erasing': 0.4,
                'crop_fraction': 1.0,
                'project': 'runs/train',
                'name': 'uatd_sonar_exp',
                'exist_ok': True,
                'pretrained': True,
                'verbose': True,
                'seed': 0,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': True,
                'close_mosaic': 10,
                'resume': False,
                'cache': False, 
                'save': True,
                'save_period': -1,
                'plots': True,
                'device': '0'
            }
        return config
    
    def setup_device(self):
        """Setup and verify GPU device"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def verify_dataset(self):
        """Verify dataset structure and files"""
        data_yaml = self.config['data_yaml']
        
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
        
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check directories
        train_dir = data_config.get('train')
        val_dir = data_config.get('val')
        
        if train_dir and os.path.exists(train_dir):
            train_images = len([f for f in os.listdir(train_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            logger.info(f"Training images: {train_images}")
        else:
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        if val_dir and os.path.exists(val_dir):
            val_images = len([f for f in os.listdir(val_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            logger.info(f"Validation images: {val_images}")
        
        logger.info(f"Number of classes: {data_config.get('nc', 'Unknown')}")
        logger.info(f"Class names: {data_config.get('names', 'Unknown')}")
        
        return data_config
    
    def initialize_model(self):
        """Initialize YOLO model"""
        print("Config:", self.config)  # Debug line

        model_path = self.config.get('model_size', 'yolo11n.pt')  # Fallback to default if missing
    
        if not os.path.exists(model_path):
            logger.info(f"Downloading pretrained model: {model_path}")
    
        self.model = YOLO(model_path)
        logger.info(f"Model initialized: {model_path}")
    
        return self.model

    
    def train(self):
        """Train the YOLO model"""
        print("DEBUG: Starting train() method")
        
        if self.model is None:
            print("DEBUG: Initializing model...")
            self.initialize_model()
            print("DEBUG: Model initialized successfully")
        
        # Verify dataset before training
        print("DEBUG: About to verify dataset...")
        try:
            self.verify_dataset()
            print("DEBUG: Dataset verification completed")
        except Exception as e:
            print(f"DEBUG: Dataset verification failed: {e}")
            raise
        
        logger.info("Starting YOLOv11 training...")
        logger.info(f"Configuration: {self.config}")
        
        try:
            print("DEBUG: Preparing training arguments...")
            # Training arguments
            train_args = {
                'data': self.config['data_yaml'],
                'epochs': self.config['epochs'],
                'imgsz': self.config['image_size'],
                'batch': self.config['batch_size'],
                'device': self.config['device'],
                'patience': self.config['patience'],
                'save_period': self.config['save_period'],
                'workers': self.config['workers'],
                'optimizer': self.config['optimizer'],
                'lr0': self.config['lr0'],
                'lrf': self.config['lrf'],
                'momentum': self.config['momentum'],
                'weight_decay': self.config['weight_decay'],
                'warmup_epochs': self.config['warmup_epochs'],
                'warmup_momentum': self.config['warmup_momentum'],
                'warmup_bias_lr': self.config['warmup_bias_lr'],
                'box': self.config['box'],
                'cls': self.config['cls'],
                'dfl': self.config['dfl'],
                'label_smoothing': self.config['label_smoothing'],
                'nbs': self.config['nbs'],
                'overlap_mask': self.config['overlap_mask'],
                'mask_ratio': self.config['mask_ratio'],
                'dropout': self.config['dropout'],
                'val': self.config['val'],
                'amp': self.config['amp'],
                'fraction': self.config['fraction'],
                'profile': self.config['profile'],
                'freeze': self.config['freeze'],
                'multi_scale': self.config['multi_scale'],
                'copy_paste': self.config['copy_paste'],
                'auto_augment': self.config['auto_augment'],
                'erasing': self.config['erasing'],
                'crop_fraction': self.config['crop_fraction'],
                'project': self.config['project'],
                'name': self.config['name'],
                'exist_ok': self.config['exist_ok'],
                'pretrained': self.config['pretrained'],
                'verbose': self.config['verbose'],
                'seed': self.config['seed'],
                'deterministic': self.config['deterministic'],
                'single_cls': self.config['single_cls'],
                'rect': self.config['rect'],
                'cos_lr': self.config['cos_lr'],
                'close_mosaic': self.config['close_mosaic'],
                'resume': self.config['resume'],
                'cache': self.config['cache'],
                'save': self.config['save'],
                'plots': self.config['plots']
            }
            
            train_args = {k: v for k, v in train_args.items() if v is not None}
            
            print("DEBUG: About to call model.train()")
            print(f"DEBUG: Training args keys: {list(train_args.keys())}")
            
            # Start training
            results = self.model.train(**train_args)
            
            print("DEBUG: model.train() completed successfully!")
            logger.info("Training completed successfully!")
            logger.info(f"Best model saved at: {self.config['project']}/{self.config['name']}/weights/best.pt")
            
            return results
            
        except Exception as e:
            print(f"DEBUG: Exception in training: {e}")
            logger.error(f"Training failed: {e}")
            raise e
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            self.initialize_model()
        
        return self.model.info()


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on UATD Sonar Dataset')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str, help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device to use')
    parser.add_argument('--name', type=str, default='uatd_sonar_exp', help='Experiment name')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOTrainer(config_path=args.config)
    
    # Override config with command line arguments
    if args.data:
        trainer.config['data_yaml'] = args.data
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch:
        trainer.config['batch_size'] = args.batch
    if args.imgsz:
        trainer.config['image_size'] = args.imgsz
    if args.device:
        trainer.config['device'] = args.device
    if args.name:
        trainer.config['name'] = args.name
    
    # Train model
    results = trainer.train()
    
    logger.info("Training session completed!")


if __name__ == "__main__":
    main()
