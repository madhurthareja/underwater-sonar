#!/usr/bin/env python3
"""
YOLOv11 Inference Script for UATD Sonar Dataset
"""

import torch
import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOPredictor:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """Initialize YOLO predictor"""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = [
            'human body', 'ball', 'cube', 'cylinder', 'model', 
            'circle cage', 'square cage', 'metal bucket', 'plane model', 'ROV'
        ]
        
    def load_model(self):
        """Load trained YOLO model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        logger.info(f"Model loaded: {self.model_path}")
        
    def predict_single_image(self, image_path, save_path=None, show=False):
        """Predict on a single image"""
        if self.model is None:
            self.load_model()
        
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device='0',
            save=save_path is not None,
            show=show,
            verbose=False
        )
        
        return results[0] if results else None
    
    def predict_batch(self, image_dir, output_dir=None, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        """Predict on a batch of images"""
        if self.model is None:
            self.load_model()
        
        # Get image paths
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(image_dir).glob(f'*{ext}'))
            image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return []
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Run batch prediction
        results = self.model.predict(
            source=image_paths,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device='0',
            save=output_dir is not None,
            project=output_dir,
            name='predictions',
            exist_ok=True,
            verbose=False
        )
        
        return results
    
    def predict_video(self, video_path, output_path=None, show=False):
        """Predict on video"""
        if self.model is None:
            self.load_model()
        
        # Run prediction on video
        results = self.model.predict(
            source=video_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device='0',
            save=output_path is not None,
            show=show,
            stream=True,
            verbose=False
        )
        
        return results
    
    def predict_webcam(self, camera_id=0, save_path=None):
        """Real-time prediction from webcam"""
        if self.model is None:
            self.load_model()
        
        # Run prediction on webcam
        results = self.model.predict(
            source=camera_id,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            device='0',
            save=save_path is not None,
            show=True,
            stream=True,
            verbose=False
        )
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """Visualize prediction results"""
        if not results:
            logger.warning("No results to visualize")
            return
        
        # Get the annotated image
        annotated_img = results.plot()
        
        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('YOLOv11 Predictions')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def extract_detections(self, results):
        """Extract detection information"""
        if not results or not hasattr(results, 'boxes'):
            return []
        
        detections = []
        boxes = results.boxes
        
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': box,
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}',
                    'area': (box[2] - box[0]) * (box[3] - box[1])
                }
                
                detections.append(detection)
        
        return detections
    
    def filter_detections(self, detections, min_conf=None, target_classes=None):
        """Filter detections based on criteria"""
        if min_conf is None:
            min_conf = self.conf_threshold
        
        filtered = []
        for det in detections:
            # Confidence filter
            if det['confidence'] < min_conf:
                continue
            
            # Class filter
            if target_classes and det['class_name'] not in target_classes:
                continue
            
            filtered.append(det)
        
        return filtered
    
    def analyze_image(self, image_path, save_analysis=False):
        """Comprehensive analysis of a single image"""
        logger.info(f"Analyzing: {image_path}")
        
        # Run prediction
        results = self.predict_single_image(image_path)
        
        if not results:
            logger.warning("No detections found")
            return
        
        # Extract detections
        detections = self.extract_detections(results)
        
        # Analysis
        analysis = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(detections),
            'detections': detections,
            'class_distribution': {},
            'confidence_stats': {}
        }
        
        if detections:
            # Class distribution
            for det in detections:
                class_name = det['class_name']
                analysis['class_distribution'][class_name] = analysis['class_distribution'].get(class_name, 0) + 1
            
            # Confidence statistics
            confidences = [det['confidence'] for det in detections]
            analysis['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        # Print analysis
        logger.info(f"=== ANALYSIS RESULTS ===")
        logger.info(f"Total detections: {analysis['total_detections']}")
        logger.info(f"Class distribution: {analysis['class_distribution']}")
        if analysis['confidence_stats']:
            stats = analysis['confidence_stats']
            logger.info(f"Confidence - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        
        # Save analysis
        if save_analysis:
            import json
            analysis_path = f"{Path(image_path).stem}_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"Analysis saved to: {analysis_path}")
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Inference for UATD Sonar Dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, required=True, help='Source: image/video/dir/webcam')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--analyze', action='store_true', help='Detailed analysis')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--device', type=str, default='0', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = YOLOPredictor(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Set device
    if torch.cuda.is_available() and args.device == '0':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Run prediction based on source type
    if args.webcam:
        # Webcam prediction
        logger.info("Starting webcam prediction...")
        results = predictor.predict_webcam(camera_id=int(args.source) if args.source.isdigit() else 0)
        
    elif os.path.isdir(args.source):
        # Directory of images
        logger.info(f"Processing directory: {args.source}")
        results = predictor.predict_batch(
            image_dir=args.source,
            output_dir=args.output if args.save else None
        )
        
        if args.analyze:
            # Analyze each image
            image_paths = list(Path(args.source).glob('*.jpg')) + list(Path(args.source).glob('*.png'))
            for img_path in image_paths[:5]:  # Analyze first 5 images
                predictor.analyze_image(img_path, save_analysis=True)
        
    elif os.path.isfile(args.source):
        # Single file (image or video)
        if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Video
            logger.info(f"Processing video: {args.source}")
            results = predictor.predict_video(
                video_path=args.source,
                output_path=args.output if args.save else None,
                show=args.show
            )
        else:
            # Image
            logger.info(f"Processing image: {args.source}")
            results = predictor.predict_single_image(
                image_path=args.source,
                save_path=args.output if args.save else None,
                show=args.show
            )
            
            if args.analyze:
                predictor.analyze_image(args.source, save_analysis=True)
            
            if results and not args.show:
                predictor.visualize_results(results, save_path=args.output)
    
    else:
        logger.error(f"Invalid source: {args.source}")
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
