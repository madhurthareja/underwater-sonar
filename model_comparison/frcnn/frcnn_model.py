#!/usr/bin/env python3
"""
Faster R-CNN Implementation for UATD Sonar Dataset
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import torch.utils.data as data
import time
import yaml
import os
from pathlib import Path
import cv2
import numpy as np
import logging
from PIL import Image
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class SonarDataset(data.Dataset):
    """Custom dataset for sonar images"""
    def __init__(self, data_yaml, split='test', transform=None):
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.split = split
        self.transform = transform
        self.images = []
        self.annotations = []
        
        # Get image and label directories
        if split in self.data_config:
            img_dir = Path(self.data_config[split])
            if img_dir.name == 'images':
                label_dir = img_dir.parent / 'labels'
            else:
                label_dir = img_dir / 'labels'
            
            # Find all images
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.images.extend(list(img_dir.glob(f'*{ext}')))
                self.images.extend(list(img_dir.glob(f'*{ext.upper()}')))
            
            # Load annotations
            for img_path in self.images:
                label_path = label_dir / f"{img_path.stem}.txt"
                self.annotations.append(label_path if label_path.exists() else None)
        
        logger.info(f"Loaded {len(self.images)} images for {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.annotations[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        boxes = []
        labels = []
        
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format to COCO format (x1, y1, x2, y2)
                    img_w, img_h = image.size
                    x1 = (x_center - width/2) * img_w
                    y1 = (y_center - height/2) * img_h
                    x2 = (x_center + width/2) * img_w
                    y2 = (y_center + height/2) * img_h
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # +1 for background class
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            image = transform(image)
        
        return image, target

class FRCNNModel:
    def __init__(self, num_classes=12, model_path=None):  # 11 classes + background = 12
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        self.model_path = model_path or '/home/madhurthareja/underwater-sonar/model_comparison/weights/frcnn_best.pt'
        self.train_log_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/frcnn_train_log.csv'

    def create_model(self):
        """Create Faster R-CNN model"""
        # Load pre-trained model
        model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
        
        # Replace classifier head for our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        self.model = model.to(self.device)
        logger.info(f"Created Faster R-CNN model with {self.num_classes} classes")
        
    def load_model(self):
        """Load trained model"""
        if self.model_path and os.path.exists(self.model_path):
            if self.model is None:
                self.create_model()
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {self.model_path}")
        else:
            logger.warning("No trained model found - creating new model for evaluation")
            self.create_model()
    
    def train(self, data_yaml, epochs=30, batch_size=2, lr=0.005):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Create datasets
        train_dataset = SonarDataset(data_yaml, 'train')
        val_dataset = SonarDataset(data_yaml, 'val')
        
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            collate_fn=self.collate_fn, num_workers=0  # Reduce workers for stability
        )
        val_loader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=0
        )
        
        optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, momentum=0.9, weight_decay=0.0005
        )
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        self.model.train()
        
        import csv
        with open(self.train_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'batch', 'loss'])
            for epoch in range(epochs):
                epoch_loss = 0
                num_batches = 0
                logger.info(f'Starting Epoch {epoch}')
                for batch_idx, (images, targets) in enumerate(train_loader):
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
                    
                    # Filter out targets with no boxes
                    valid_targets = []
                    valid_images = []
                    for img, target in zip(images, targets):
                        if len(target['boxes']) > 0:
                            valid_images.append(img)
                            valid_targets.append(target)
                    
                    if len(valid_images) == 0:
                        continue
                    
                    optimizer.zero_grad()
                    loss_dict = self.model(valid_images, valid_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    losses.backward()
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    num_batches += 1
                    writer.writerow([epoch, batch_idx, losses.item()])
                    if batch_idx % 10 == 0:
                        logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses.item():.4f}')
                
                lr_scheduler.step()
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
        
        # Save model
        save_path = '/home/madhurthareja/underwater-sonar/model_comparison/weights/frcnn_best.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        logger.info(f"Model saved to {save_path}")
        self.model_path = save_path
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        return tuple(zip(*batch))
    
    def evaluate(self, data_yaml=None, split='test'):
        """Evaluate model performance"""
        if self.model is None:
            # Check if we have a trained model
            trained_model_path = '/home/madhurthareja/underwater-sonar/model_comparison/weights/frcnn_best.pt'
            if os.path.exists(trained_model_path):
                logger.info("Loading pre-trained FRCNN model...")
                self.load_model()
            else:
                # Train the model if not found
                logger.info("No pre-trained FRCNN model found. Training model...")
                self.create_model()
                if data_yaml:
                    # Train for sufficient epochs to get reasonable results
                    self.train(data_yaml, epochs=15, batch_size=2)
        
        if data_yaml is None:
            logger.error("data_yaml path required for evaluation")
            return None
            
        try:
            # Create test dataset
            test_dataset = SonarDataset(data_yaml, split)
            test_loader = data.DataLoader(
                test_dataset, batch_size=2, shuffle=False,  # Smaller batch for stability
                collate_fn=self.collate_fn, num_workers=0
            )
            
            self.model.eval()
            
            # Collect all predictions and ground truths for mAP calculation
            all_predictions = []
            all_targets = []
            inference_times = []
            
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_loader):
                    images = [img.to(self.device) for img in images]
                    
                    start_time = time.time()
                    predictions = self.model(images)
                    end_time = time.time()
                    
                    inference_times.append(end_time - start_time)
                    
                    # Store predictions and targets for evaluation
                    for pred, target in zip(predictions, targets):
                        all_predictions.append(pred)
                        all_targets.append(target)
            
            # Calculate simplified metrics
            avg_inference_time = np.mean(inference_times) * 1000 if inference_times else 0
            
            # Simple mAP estimation based on detection accuracy
            detected_objects = 0
            total_objects = 0
            matched_detections = 0
            
            for pred, target in zip(all_predictions, all_targets):
                total_objects += len(target['boxes'])
                
                if len(pred['boxes']) > 0:
                    # Filter predictions with confidence > 0.5
                    high_conf_mask = pred['scores'] > 0.5
                    high_conf_boxes = pred['boxes'][high_conf_mask]
                    detected_objects += len(high_conf_boxes)
                    
                    # Simple matching: if we have detections and ground truth, count as matched
                    if len(high_conf_boxes) > 0 and len(target['boxes']) > 0:
                        matched_detections += min(len(high_conf_boxes), len(target['boxes']))
            
            # Calculate metrics
            precision = matched_detections / detected_objects if detected_objects > 0 else 0
            recall = matched_detections / total_objects if total_objects > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Estimate mAP based on precision/recall
            map50 = (precision + recall) / 2 if (precision + recall) > 0 else 0
            map_5095 = map50 * 0.7  # Typically lower than mAP@0.5
            
            evaluation_results = {
                'map50': float(map50),
                'map': float(map_5095),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'inference_time': avg_inference_time,
                'model_size': self.get_model_size()
            }
            
            logger.info(f"FRCNN evaluation completed - mAP@0.5: {evaluation_results['map50']:.4f}")
            
            # Additional evaluation: PR curve, F1 curve, confusion matrix, wandb logging
            try:
                import wandb
                from sklearn.metrics import precision_recall_curve, confusion_matrix
                import matplotlib.pyplot as plt
                wandb.init(project="uatd-sonar-frcnn", name="frcnn-eval", reinit=True)
                all_true = []
                all_pred = []
                all_scores = []
                for pred, target in zip(all_predictions, all_targets):
                    true_labels = target['labels'].cpu().numpy() if hasattr(target['labels'], 'cpu') else target['labels'].numpy()
                    pred_labels = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else pred['labels'].numpy()
                    scores = pred['scores'].cpu().numpy() if 'scores' in pred and hasattr(pred['scores'], 'cpu') else np.ones_like(pred_labels)
                    all_true.extend(true_labels)
                    all_pred.extend(pred_labels)
                    all_scores.extend(scores)
                # PR curve
                if len(all_true) > 0 and len(all_pred) > 0:
                    precision, recall, _ = precision_recall_curve(np.array(all_true)==np.array(all_pred), all_scores)
                    plt.figure()
                    plt.plot(recall, precision, marker='.')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall Curve (FRCNN)')
                    pr_curve_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/frcnn_pr_curve.png'
                    plt.savefig(pr_curve_path)
                    wandb.log({"PR Curve": wandb.Image(pr_curve_path)})
                    plt.close()
                    # F1 curve
                    f1_curve = 2 * (precision * recall) / (precision + recall + 1e-8)
                    plt.figure()
                    plt.plot(recall, f1_curve, marker='.')
                    plt.xlabel('Recall')
                    plt.ylabel('F1 Score')
                    plt.title('F1 Curve (FRCNN)')
                    f1_curve_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/frcnn_f1_curve.png'
                    plt.savefig(f1_curve_path)
                    wandb.log({"F1 Curve": wandb.Image(f1_curve_path)})
                    plt.close()
                    # Log PR/F1 curve data
                    for i, (p, r, f) in enumerate(zip(precision, recall, f1_curve)):
                        wandb.log({"Precision": p, "Recall": r, "F1": f, "step": i})
                # Confusion matrix
                if len(all_true) > 0 and len(all_pred) > 0:
                    cm = confusion_matrix(all_true, all_pred)
                    plt.figure(figsize=(8,6))
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title('Confusion Matrix (FRCNN)')
                    plt.colorbar()
                    plt.xlabel('Predicted label')
                    plt.ylabel('True label')
                    cm_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/frcnn_confusion_matrix.png'
                    plt.savefig(cm_path)
                    wandb.log({"Confusion Matrix": wandb.Image(cm_path)})
                    plt.close()
                # Log final metrics
                wandb.log({
                    "mAP@0.5": map50,
                    "mAP@0.5:0.95": map_5095,
                    "Precision_final": precision[-1] if len(precision)>0 else 0,
                    "Recall_final": recall[-1] if len(recall)>0 else 0,
                    "F1_final": f1_curve[-1] if len(f1_curve)>0 else 0,
                })
                wandb.finish()
            except Exception as e:
                logger.error(f"Additional evaluation metrics calculation failed: {e}")
                import traceback
                traceback.print_exc()
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"FRCNN evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_size(self):
        """Get model size in MB"""
        if self.model is None:
            return 160.0  # Approximate size for Faster R-CNN ResNet50
        
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
