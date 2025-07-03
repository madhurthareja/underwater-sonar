#!/usr/bin/env python3
"""
Utility functions for YOLO training and data processing
"""

import os
import cv2
import numpy as np
import yaml
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
import torch
import pandas as pd
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetUtils:
    """Utilities for dataset management and analysis"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.class_names = [
            'human body', 'ball', 'cube', 'cylinder', 'model', 'tyre', 'circle cage',
            'square cage', 'metal bucket', 'plane model', 'plane', 'ROV', 'rov'
        ]
        self.class_mapping = {
            'human body': 0,
            'ball': 1,
            'cube': 2,
            'cylinder': 3,
            'model': 4,
            'tyre': 5,
            'circle cage': 6,
            'square cage': 7,
            'metal bucket': 8,
            'plane': 9,
            'rov': 10
        }
    
    def convert_xml_to_yolo(self, xml_dir, output_dir, img_dir=None):
        """Convert XML annotations to YOLO format"""
        xml_dir = Path(xml_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        xml_files = list(xml_dir.glob('*.xml'))
        logger.info(f"Converting {len(xml_files)} XML files to YOLO format...")
        
        converted = 0
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get image dimensions
                size = root.find('size')
                if size is None:
                    logger.warning(f"No size info in {xml_file}, skipping...")
                    continue
                
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                
                # Process objects
                yolo_lines = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in self.class_mapping:
                        logger.warning(f"Unknown class '{class_name}' in {xml_file}")
                        continue
                    
                    class_id = self.class_mapping[class_name]
                    
                    # Get bounding box
                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        continue
                    
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # Convert to YOLO format (normalized)
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    # Validate coordinates
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    else:
                        logger.warning(f"Invalid coordinates in {xml_file}: {class_name}")
                
                # Write YOLO file
                if yolo_lines:
                    output_txt = output_dir / f"{xml_file.stem}.txt"
                    with open(output_txt, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    converted += 1
                
            except Exception as e:
                logger.error(f"Error converting {xml_file}: {e}")
        
        logger.info(f"Successfully converted {converted} files")
        return converted
    
    def analyze_dataset(self, data_yaml_path):
        """Analyze dataset statistics"""
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        stats = {
            'splits': {},
            'class_distribution': Counter(),
            'bbox_stats': [],
            'image_stats': []
        }
        
        # Analyze each split
        for split in ['train', 'val', 'test']:
            if split in data_config:
                img_dir = Path(data_config[split])
                label_dir = img_dir.parent / 'labels'
                
                split_stats = self._analyze_split(img_dir, label_dir)
                stats['splits'][split] = split_stats
                
                # Aggregate class distribution
                stats['class_distribution'].update(split_stats['class_distribution'])
                stats['bbox_stats'].extend(split_stats['bbox_stats'])
                stats['image_stats'].extend(split_stats['image_stats'])
        
        return stats
    
    def _analyze_split(self, img_dir, label_dir):
        """Analyze a single dataset split"""
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_files = []
        for ext in img_extensions:
            img_files.extend(img_dir.glob(f'*{ext}'))
            img_files.extend(img_dir.glob(f'*{ext.upper()}'))
        
        split_stats = {
            'num_images': len(img_files),
            'num_labels': 0,
            'class_distribution': Counter(),
            'bbox_stats': [],
            'image_stats': []
        }
        
        for img_file in img_files:
            label_file = label_dir / f"{img_file.stem}.txt"
            
            # Image statistics
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    split_stats['image_stats'].append({'width': w, 'height': h, 'aspect_ratio': w/h})
            except:
                pass
            
            # Label statistics
            if label_file.exists():
                split_stats['num_labels'] += 1
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                            split_stats['class_distribution'][class_name] += 1
                            split_stats['bbox_stats'].append({
                                'class': class_name,
                                'width': width,
                                'height': height,
                                'area': width * height
                            })
        
        return split_stats
    
    def visualize_dataset_stats(self, stats, save_dir='dataset_analysis'):
        """Create visualizations of dataset statistics"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Class distribution
        plt.figure(figsize=(12, 8))
        classes = list(stats['class_distribution'].keys())
        counts = list(stats['class_distribution'].values())
        
        plt.subplot(2, 2, 1)
        plt.bar(classes, counts)
        plt.title('Class Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        
        # Bounding box size distribution
        if stats['bbox_stats']:
            widths = [bbox['width'] for bbox in stats['bbox_stats']]
            heights = [bbox['height'] for bbox in stats['bbox_stats']]
            areas = [bbox['area'] for bbox in stats['bbox_stats']]
            
            plt.subplot(2, 2, 2)
            plt.hist(areas, bins=50, alpha=0.7)
            plt.title('Bounding Box Area Distribution')
            plt.xlabel('Area (normalized)')
            plt.ylabel('Count')
            
            plt.subplot(2, 2, 3)
            plt.scatter(widths, heights, alpha=0.5)
            plt.title('Bounding Box Width vs Height')
            plt.xlabel('Width (normalized)')
            plt.ylabel('Height (normalized)')
            
        # Image aspect ratios
        if stats['image_stats']:
            aspect_ratios = [img['aspect_ratio'] for img in stats['image_stats']]
            
            plt.subplot(2, 2, 4)
            plt.hist(aspect_ratios, bins=30, alpha=0.7)
            plt.title('Image Aspect Ratio Distribution')
            plt.xlabel('Aspect Ratio (W/H)')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/dataset_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Dataset analysis plots saved to {save_dir}")
    
    def create_data_yaml(self, train_dir, val_dir, test_dir, save_path):
        """Create data.yaml configuration file"""
        config = {
            'train': str(Path(train_dir).absolute()),
            'val': str(Path(val_dir).absolute()),
            'test': str(Path(test_dir).absolute()),
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Data config saved to: {save_path}")
        return save_path
    
    def split_dataset(self, source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_files = []
        for ext in img_extensions:
            img_files.extend(source_dir.glob(f'**/*{ext}'))
            img_files.extend(source_dir.glob(f'**/*{ext.upper()}'))
        
        # Shuffle and split
        np.random.shuffle(img_files)
        n_total = len(img_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': img_files[:n_train],
            'val': img_files[n_train:n_train + n_val],
            'test': img_files[n_train + n_val:]
        }
        
        # Copy files
        for split, files in splits.items():
            logger.info(f"Copying {len(files)} files to {split} set...")
            
            for img_file in files:
                # Copy image
                dst_img = output_dir / split / 'images' / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding label if exists
                label_file = img_file.parent.parent / 'labels' / f"{img_file.stem}.txt"
                if label_file.exists():
                    dst_label = output_dir / split / 'labels' / f"{img_file.stem}.txt"
                    shutil.copy2(label_file, dst_label)
        
        logger.info("Dataset split completed!")
        return splits


class TrainingUtils:
    """Utilities for training monitoring and optimization"""
    
    @staticmethod
    def monitor_gpu_memory():
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'total': memory_total,
                'free': memory_total - memory_reserved
            }
        return None
    
    @staticmethod
    def plot_training_curves(results_csv, save_path=None):
        """Plot training curves from results.csv"""
        df = pd.read_csv(results_csv)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training losses
        if 'train/box_loss' in df.columns:
            axes[0,0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
            axes[0,0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
            axes[0,0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
            axes[0,0].set_title('Training Losses')
            axes[0,0].legend()
            axes[0,0].grid(True)
        
        # Validation losses
        if 'val/box_loss' in df.columns:
            axes[0,1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            axes[0,1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
            axes[0,1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
            axes[0,1].set_title('Validation Losses')
            axes[0,1].legend()
            axes[0,1].grid(True)
        
        # mAP scores
        if 'metrics/mAP50(B)' in df.columns:
            axes[1,0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
            axes[1,0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
            axes[1,0].set_title('mAP Scores')
            axes[1,0].legend()
            axes[1,0].grid(True)
        
        # Precision and recall
        if 'metrics/precision(B)' in df.columns:
            axes[1,1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
            axes[1,1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
            axes[1,1].set_title('Precision & Recall')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def optimize_batch_size(model_path, data_yaml, max_batch=32, device='0'):
        """Find optimal batch size for available GPU memory"""
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        optimal_batch = 1
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            if batch_size > max_batch:
                break
            
            try:
                logger.info(f"Testing batch size: {batch_size}")
                torch.cuda.empty_cache()
                
                # Test training for 1 epoch
                model.train(
                    data=data_yaml,
                    epochs=1,
                    batch=batch_size,
                    device=device,
                    verbose=False,
                    save=False
                )
                
                optimal_batch = batch_size
                logger.info(f"Batch size {batch_size} OK")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Batch size {batch_size} failed: {e}")
                    break
                else:
                    raise e
        
        logger.info(f"Optimal batch size: {optimal_batch}")
        return optimal_batch


def main():
    """Main function for utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Dataset and Training Utilities')
    parser.add_argument('--convert-xml', type=str, help='Convert XML annotations to YOLO format')
    parser.add_argument('--analyze-dataset', type=str, help='Analyze dataset statistics')
    parser.add_argument('--split-dataset', type=str, help='Split dataset into train/val/test')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--data-yaml', type=str, help='Path to data.yaml file')
    
    args = parser.parse_args()
    
    if args.convert_xml:
        utils = DatasetUtils(args.convert_xml)
        utils.convert_xml_to_yolo(
            xml_dir=f"{args.convert_xml}/annotations",
            output_dir=f"{args.output}/labels"
        )
    
    if args.analyze_dataset and args.data_yaml:
        utils = DatasetUtils(args.analyze_dataset)
        stats = utils.analyze_dataset(args.data_yaml)
        utils.visualize_dataset_stats(stats, save_dir=args.output)
    
    if args.split_dataset:
        utils = DatasetUtils(args.split_dataset)
        utils.split_dataset(
            source_dir=args.split_dataset,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
