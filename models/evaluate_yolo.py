#!/usr/bin/env python3
"""
YOLOv8n Evaluation Script for UATD Sonar Dataset
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
                ap_per_class = results.box.ap  # Already 1D: AP@0.5 for each class
                
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
            f.write("YOLOv8n UATD Sonar Dataset Evaluation Report\n")
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
    
    def create_visual_summary(self, save_path='evaluation_summary.png'):
        """Create and save a visual summary of evaluation results"""
        if not self.results:
            logger.warning("No evaluation results available. Run evaluation first.")
            return
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLOv8n UATD Sonar Dataset Evaluation Summary', fontsize=16, fontweight='bold')
        
        # Color scheme
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Get results for the first available split
        split_name = list(self.results.keys())[0]
        results = self.results[split_name]
        
        if hasattr(results, 'box'):
            metrics = results.box
            
            # 1. Overall Metrics Bar Chart
            ax1 = axes[0, 0]
            metric_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
            metric_values = [metrics.map50, metrics.map, metrics.mp, metrics.mr]
            
            bars = ax1.bar(metric_names, metric_values, color=colors)
            ax1.set_title('Overall Performance Metrics', fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Class-wise Performance
            ax2 = axes[0, 1]
            if hasattr(results.box, 'ap') and hasattr(results.box, 'ap_class_index'):
                # Get class names
                with open(self.data_yaml, 'r') as f:
                    data_config = yaml.safe_load(f)
                class_names = data_config.get('names', [])
                
                ap_per_class = results.box.ap
                
                # Limit to top 10 classes for readability
                if len(class_names) > 10:
                    top_indices = np.argsort(ap_per_class)[-10:]
                    class_names = [class_names[i] for i in top_indices]
                    ap_per_class = ap_per_class[top_indices]
                
                bars = ax2.barh(class_names, ap_per_class, color=colors[1])
                ax2.set_title('Class-wise AP@0.5', fontweight='bold')
                ax2.set_xlabel('AP@0.5')
                ax2.set_xlim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, ap_per_class):
                    width = bar.get_width()
                    ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                            f'{value:.3f}', ha='left', va='center', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'Class-wise data not available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Class-wise AP@0.5', fontweight='bold')
        
        # 3. Model Information
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # Model specs from your training log
        model_info = [
            f"Model: YOLOv8n (Custom)",
            f"Parameters: 3.01M",
            f"GFLOPs: 8.2",
            f"Layers: 225",
            f"Input Size: 640x640",
            f"Dataset: {os.path.basename(self.data_yaml)}",
            f"Evaluated on: {split_name.upper()}"
        ]
        
        info_text = '\n'.join(model_info)
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor=colors[2], alpha=0.3))
        ax3.set_title('Model Information', fontweight='bold')
        
        # 4. Performance Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if hasattr(results, 'box'):
            f1_score = 2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr)
            
            summary_text = [
                f"🎯 mAP@0.5: {metrics.map50:.4f}",
                f"🎯 mAP@0.5:0.95: {metrics.map:.4f}",
                f"🎯 Precision: {metrics.mp:.4f}",
                f"🎯 Recall: {metrics.mr:.4f}",
                f"🎯 F1-Score: {f1_score:.4f}",
                "",
                f"📊 Performance Grade: {'Excellent' if metrics.map50 > 0.9 else 'Good' if metrics.map50 > 0.7 else 'Fair'}",
                f"⚡ Model Size: Nano (3.01M params)",
                f"🚀 Inference: Real-time capable"
            ]
            
            summary_text_str = '\n'.join(summary_text)
            ax4.text(0.1, 0.9, summary_text_str, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor=colors[3], alpha=0.3))
        
        ax4.set_title('Performance Summary', fontweight='bold')
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', 
                fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visual summary saved to: {save_path}")
        return save_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8n on UATD Sonar Dataset')
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
    
    # Create visual summary
    evaluator.create_visual_summary()
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
