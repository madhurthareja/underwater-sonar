#!/usr/bin/env python3
"""
YOLOv11 Evaluation Script for UATD Sonar Dataset
"""

import torch
import os
import argparse
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOEvaluator:
    def __init__(self, model_path, data_yaml):
        """Initialize YOLO evaluator"""
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = None
        self.results = {}
        
    def load_model(self):
        """Load trained YOLO model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        if self.model is None:
            raise RuntimeError(f"Failed to load YOLO model from: {self.model_path}")
        logger.info(f"Model loaded: {self.model_path}")
        
    def evaluate_on_test_set(self, test_split='test', imgsz=640, batch=8, device='0'):
        """Evaluate model on test dataset"""
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded. Cannot evaluate.")
        logger.info(f"Evaluating on {test_split} split...")
        
        # Run validation on test set
        results = self.model.val(
            data=self.data_yaml,
            split=test_split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            plots=True
        )
        
        self.results[test_split] = results
        
        # Print metrics
        if hasattr(results, 'box'):
            metrics = results.box
            logger.info(f"=== {test_split.upper()} SET RESULTS ===")
            logger.info(f"mAP@0.5: {metrics.map50:.4f}")
            logger.info(f"mAP@0.5:0.95: {metrics.map:.4f}")
            logger.info(f"Precision: {metrics.mp:.4f}")
            logger.info(f"Recall: {metrics.mr:.4f}")
            logger.info(f"F1-Score: {2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr):.4f}")
        
        return results
    
    def evaluate_all_splits(self):
        """Evaluate on all available splits"""
        splits = ['val', 'test']
        all_results = {}
        
        for split in splits:
            try:
                results = self.evaluate_on_test_set(test_split=split)
                all_results[split] = results
            except Exception as e:
                logger.warning(f"Could not evaluate on {split} split: {e}")
        
        return all_results
    
    def class_wise_analysis(self, split='test'):
        """Perform class-wise performance analysis"""
        if split not in self.results:
            self.evaluate_on_test_set(test_split=split)
        
        results = self.results[split]
        
        if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
            # Get class names
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get('names', [])
            
            # Class-wise AP
            if hasattr(results.box, 'ap'):
                ap_per_class = results.box.ap[:, 0]  # AP@0.5
                
                logger.info(f"=== CLASS-WISE PERFORMANCE ({split.upper()}) ===")
                for i, (class_name, ap) in enumerate(zip(class_names, ap_per_class)):
                    logger.info(f"{class_name}: AP@0.5 = {ap:.4f}")
        
        return results
    
    def generate_confusion_matrix(self, split='test', save_dir='runs/evaluate'):
        """Generate and save confusion matrix"""
        if self.model is None:
            self.load_model()
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Run evaluation with confusion matrix
        results = self.model.val(
            data=self.data_yaml,
            split=split,
            save_json=True,
            plots=True,
            project=save_dir,
            name=f'confusion_matrix_{split}'
        )
        
        logger.info(f"Confusion matrix saved to: {save_dir}/confusion_matrix_{split}")
        return results
    
    def benchmark_inference_speed(self, num_images=100, imgsz=640):
        """Benchmark inference speed"""
        if self.model is None:
            self.load_model()
        
        # Load test images
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        test_dir = data_config.get('test')
        if not test_dir or not os.path.exists(test_dir):
            logger.warning("Test directory not found for speed benchmark")
            return
        
        # Get image paths
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(Path(test_dir).glob(f'*{ext}'))
            img_paths.extend(Path(test_dir).glob(f'*{ext.upper()}'))
        
        img_paths = img_paths[:num_images]
        
        if not img_paths:
            logger.warning("No test images found for speed benchmark")
            return
        
        # Benchmark
        logger.info(f"Benchmarking inference speed on {len(img_paths)} images...")
        
        # Warmup
        for _ in range(10):
            self.model.predict(img_paths[0], imgsz=imgsz, device='0', verbose=False)
        
        # Actual benchmark
        start_time = datetime.now()
        
        for img_path in img_paths:
            self.model.predict(img_path, imgsz=imgsz, device='0', verbose=False)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        fps = len(img_paths) / total_time
        ms_per_image = (total_time * 1000) / len(img_paths)
        
        logger.info(f"=== INFERENCE SPEED BENCHMARK ===")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"FPS: {fps:.2f}")
        logger.info(f"ms per image: {ms_per_image:.2f}")
        
        return {
            'fps': fps,
            'ms_per_image': ms_per_image,
            'total_time': total_time,
            'num_images': len(img_paths)
        }
    
    def export_model(self, export_format='onnx', save_dir='runs/export'):
        """Export model to different formats"""
        if self.model is None:
            self.load_model()
        
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Exporting model to {export_format} format...")
        
        try:
            export_path = self.model.export(
                format=export_format,
                imgsz=640,
                half=True,
                dynamic=False,
                simplify=True,
                opset=11
            )
            
            logger.info(f"Model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    def generate_report(self, save_path='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        with open(save_path, 'w') as f:
            f.write("YOLOv11 UATD Sonar Dataset Evaluation Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.data_yaml}\n\n")
            
            for split, results in self.results.items():
                f.write(f"{split.upper()} SET RESULTS:\n")
                f.write("-" * 20 + "\n")
                
                if hasattr(results, 'box'):
                    metrics = results.box
                    f.write(f"mAP@0.5: {metrics.map50:.4f}\n")
                    f.write(f"mAP@0.5:0.95: {metrics.map:.4f}\n")
                    f.write(f"Precision: {metrics.mp:.4f}\n")
                    f.write(f"Recall: {metrics.mr:.4f}\n")
                    f.write(f"F1-Score: {2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr):.4f}\n")
                f.write("\n")
        
        logger.info(f"Evaluation report saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 on UATD Sonar Dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='Device to use')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark')
    parser.add_argument('--export', type=str, help='Export format (onnx, tensorrt, etc.)')
    parser.add_argument('--all-splits', action='store_true', help='Evaluate on all splits')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = YOLOEvaluator(args.model, args.data)
    
    # Run evaluation
    if args.all_splits:
        evaluator.evaluate_all_splits()
    else:
        evaluator.evaluate_on_test_set(
            test_split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
    
    # Class-wise analysis
    evaluator.class_wise_analysis(split=args.split)
    
    # Generate confusion matrix
    evaluator.generate_confusion_matrix(split=args.split)
    
    # Speed benchmark
    if args.benchmark:
        evaluator.benchmark_inference_speed()
    
    # Export model
    if args.export:
        evaluator.export_model(export_format=args.export)
    
    # Generate report
    evaluator.generate_report()
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
