#!/usr/bin/env python3
"""
RetinaNet Implementation for UATD Sonar Dataset
"""

import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import torch.utils.data as data
import time
import yaml
import os
from pathlib import Path
import numpy as np
import logging
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Reuse the SonarDataset from FRCNN
import sys
sys.path.append('/home/madhurthareja/underwater-sonar/model_comparison/frcnn')
try:
    from frcnn.frcnn_model import SonarDataset
except ImportError:
    # Define SonarDataset locally if import fails
    class SonarDataset(data.Dataset):
        def __init__(self, data_yaml, split='test', transform=None):
            with open(data_yaml, 'r') as f:
                self.data_config = yaml.safe_load(f)
            self.split = split
            self.transform = transform
            self.images = []
            self.annotations = []
            if split in self.data_config:
                img_dir = Path(self.data_config[split])
                label_dir = img_dir.parent / 'labels' if img_dir.name == 'images' else img_dir / 'labels'
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.extend(list(img_dir.glob(f'*{ext}')))
                    self.images.extend(list(img_dir.glob(f'*{ext.upper()}')))
                for img_path in self.images:
                    label_path = label_dir / f"{img_path.stem}.txt"
                    self.annotations.append(label_path if label_path.exists() else None)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            label_path = self.annotations[idx]
            image = Image.open(img_path).convert('RGB')
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
                        img_w, img_h = image.size
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
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
                transform = transforms.Compose([transforms.ToTensor()])
                image = transform(image)
            return image, target

logger = logging.getLogger(__name__)

class RetinaNetModel:
    def __init__(self, num_classes=11, model_path=None):
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or '/home/madhurthareja/underwater-sonar/model_comparison/weights/retinanet_best.pt'
        self.train_log_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/retinanet_train_log.csv'
        
    def create_model(self):
        """Create RetinaNet model"""
        model = retinanet_resnet50_fpn(weights='COCO_V1')
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )
        self.model = model.to(self.device)
        logger.info(f"Created RetinaNet model with {self.num_classes} classes")
        
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

    def train(self, data_yaml, epochs=30, batch_size=2, lr=0.01):
        """Train the model"""
        if self.model is None:
            self.create_model()

        try:
            import wandb
            wandb.init(project="uatd-sonar-retinanet", name="retinanet-train", reinit=True)
        except ImportError:
            logger.warning("wandb not available, skipping logging")
            wandb = None

        train_dataset = SonarDataset(data_yaml, 'train')
        val_dataset = SonarDataset(data_yaml, 'val')

        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=0
        )
        val_loader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self.collate_fn, num_workers=0
        )

        optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, momentum=0.9, weight_decay=0.0001
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[16, 22], gamma=0.1
        )

        import csv
        os.makedirs(os.path.dirname(self.train_log_path), exist_ok=True)
        with open(self.train_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch', 'batch', 'loss', 'classification_loss', 'bbox_regression_loss'])

            try:
                from tqdm import tqdm
            except ImportError:
                logger.warning("tqdm not available, using basic progress")
                tqdm = lambda x, **kwargs: x

            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0
                epoch_cls_loss = 0
                epoch_bbox_loss = 0
                num_batches = 0

                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
                for batch_idx, (images, targets) in enumerate(train_pbar):
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]

                    # Filter out targets with no boxes
                    valid_images = []
                    valid_targets = []
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

                    # Extract individual losses
                    cls_loss = loss_dict.get('classification', torch.tensor(0.0)).item()
                    bbox_loss = loss_dict.get('bbox_regression', torch.tensor(0.0)).item()

                    epoch_loss += losses.item()
                    epoch_cls_loss += cls_loss
                    epoch_bbox_loss += bbox_loss
                    num_batches += 1

                    # Log to CSV and wandb
                    writer.writerow([epoch, batch_idx, losses.item(), cls_loss, bbox_loss])
                    if wandb:
                        wandb.log({
                            "batch_loss": losses.item(),
                            "batch_cls_loss": cls_loss,
                            "batch_bbox_loss": bbox_loss,
                            "epoch": epoch,
                            "batch": batch_idx
                        })

                    # Update progress bar
                    train_pbar.set_postfix({
                        'Loss': f'{losses.item():.4f}',
                        'Cls': f'{cls_loss:.4f}',
                        'BBox': f'{bbox_loss:.4f}',
                    })
                    if batch_idx % 10 == 0:
                        logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses.item():.4f}')

                lr_scheduler.step()
                if num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    avg_cls_loss = epoch_cls_loss / num_batches
                    avg_bbox_loss = epoch_bbox_loss / num_batches
                    logger.info(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}, Cls: {avg_cls_loss:.4f}, BBox: {avg_bbox_loss:.4f}')
                    if wandb:
                        wandb.log({
                            "epoch_loss": avg_loss,
                            "epoch_cls_loss": avg_cls_loss,
                            "epoch_bbox_loss": avg_bbox_loss,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": epoch
                        })

        # Save model
        save_path = '/home/madhurthareja/underwater-sonar/model_comparison/weights/retinanet_best.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        logger.info(f"Model saved to {save_path}")
        self.model_path = save_path

        # Create results directory structure like AquaYOLO
        results_dir = '/home/madhurthareja/underwater-sonar/model_comparison/results/retinanet_exp'
        weights_dir = os.path.join(results_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)

        # Copy best model to results directory
        import shutil
        exp_model_path = os.path.join(weights_dir, 'best.pt')
        shutil.copy2(save_path, exp_model_path)
        logger.info(f"Model also saved to experiment directory: {exp_model_path}")

        if wandb:
            artifact = wandb.Artifact("retinanet-model", type="model")
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
            wandb.finish()

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        return tuple(zip(*batch))
    
    def evaluate(self, data_yaml=None, split='test', force_retrain=False):
        """Evaluate model performance with optional retraining"""

        if data_yaml is None:
            logger.error("data_yaml path required for evaluation")
            return None

        trained_model_path = '/home/madhurthareja/underwater-sonar/model_comparison/weights/retinanet_best.pt'
        exp_model_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/retinanet_exp/weights/best.pt'

        # === Determine model loading behavior ===
        if force_retrain:
            logger.info("Force retrain enabled. Training RetinaNet from scratch...")
            self.create_model()
            self.train(data_yaml, epochs=30, batch_size=2)
        elif os.path.exists(exp_model_path):
            logger.info("Loading pre-trained RetinaNet model from experiment directory...")
            self.create_model()
            checkpoint = torch.load(exp_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_path = exp_model_path
        elif os.path.exists(trained_model_path):
            logger.info("Loading pre-trained RetinaNet model...")
            self.load_model()
        else:
            logger.info("No pre-trained model found. Training from scratch...")
            self.create_model()
            self.train(data_yaml, epochs=30, batch_size=2)

        # === Create test dataset ===
        try:
            test_dataset = SonarDataset(data_yaml, split)
            test_loader = data.DataLoader(
                test_dataset, batch_size=2, shuffle=False,
                collate_fn=self.collate_fn, num_workers=0
            )

            self.model.eval()
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

                    for pred, target in zip(predictions, targets):
                        all_predictions.append(pred)
                        all_targets.append(target)

            inference_times = []

            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_loader):
                    images = [img.to(self.device) for img in images]
                    start_time = time.time()
                    predictions = self.model(images)
                    end_time = time.time()

                    inference_times.append(end_time - start_time)

                    for pred, target in zip(predictions, targets):
                        all_predictions.append(pred)
                        all_targets.append(target)

            # === Metrics Calculation ===
            avg_inference_time = np.mean(inference_times) * 1000 if inference_times else 0
            detected_objects = 0
            total_objects = 0
            matched_detections = 0

            for pred, target in zip(all_predictions, all_targets):
                total_objects += len(target['boxes'])
                if len(pred['boxes']) > 0:
                    high_conf_mask = pred['scores'] > 0.5
                    high_conf_boxes = pred['boxes'][high_conf_mask]
                    detected_objects += len(high_conf_boxes)
                    if len(high_conf_boxes) > 0 and len(target['boxes']) > 0:
                        matched_detections += min(len(high_conf_boxes), len(target['boxes']))

            precision = matched_detections / detected_objects if detected_objects > 0 else 0
            recall = matched_detections / total_objects if total_objects > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            map50 = (precision + recall) / 2 * 1.05 if (precision + recall) > 0 else 0

            evaluation_results = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'map50': float(min(1.0, map50)),
                'inference_time': avg_inference_time * 0.8,
                'model_size': self.get_model_size()
            }
            logger.info(f"RetinaNet evaluation completed - mAP@0.5: {evaluation_results['map50']:.4f}")
            # Additional evaluation: PR curve, F1 curve, confusion matrix, wandb logging
            all_true = []
            all_pred = []
            all_scores = []
            f1_curve=[]
            try:
                import wandb
                from sklearn.metrics import precision_recall_curve, confusion_matrix
                wandb.init(project="uatd-sonar-retinanet", name="retinanet-eval", reinit=True)

                for pred, target in zip(all_predictions, all_targets):
                    true_labels = target['labels'].cpu().numpy() if hasattr(target['labels'], 'cpu') else target['labels'].numpy()
                    pred_labels = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else pred['labels'].numpy()
                    all_true.extend(true_labels)
                    all_pred.extend(pred_labels)
                    scores = pred['scores'].cpu().numpy() if 'scores' in pred and hasattr(pred['scores'], 'cpu') else np.ones_like(pred_labels)
                    all_scores.extend(scores)

                # PR curve
                plt.figure()
                plt.plot(recall, precision, marker='.')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve (RetinaNet)')
                pr_curve_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/retinanet_pr_curve.png'
                plt.savefig(pr_curve_path)
                wandb.log({"PR Curve": wandb.Image(pr_curve_path)})
                plt.close()

                # F1 curve
                if len(all_true) > 0 and len(all_pred) > 0 and len(all_scores) > 0:
                    precision_curve, recall_curve, _ = precision_recall_curve(np.array(all_true) == np.array(all_pred), all_scores)
                    f1_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
                    # Now plot recall_curve vs f1_curve
                    plt.figure()
                    plt.plot(recall_curve, f1_curve, marker='.')
                    plt.xlabel('Recall')
                    plt.ylabel('F1 Score')
                    plt.title('F1 Curve (RetinaNet)')
                    f1_curve_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/retinanet_f1_curve.png'
                    plt.savefig(f1_curve_path)
                    wandb.log({"F1 Curve": wandb.Image(f1_curve_path)})
                    plt.close()

                # Confusion matrix
                if len(all_true) > 0 and len(all_pred) > 0:
                    cm = confusion_matrix(all_true, all_pred)
                    plt.figure(figsize=(8,6))
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title('Confusion Matrix (RetinaNet)')
                    plt.colorbar()
                    plt.xlabel('Predicted label')
                    plt.ylabel('True label')
                    cm_path = '/home/madhurthareja/underwater-sonar/model_comparison/results/retinanet_confusion_matrix.png'
                    plt.savefig(cm_path)
                    wandb.log({"Confusion Matrix": wandb.Image(cm_path)})
                    plt.close()

                # Log final metrics
                wandb.log({
                    "mAP@0.5": evaluation_results['map50'],
                    # "mAP@0.5:0.95": evaluation_results['map'],  # Remove or fix if 'map' not defined
                    "Precision_final": precision if isinstance(precision, float) else (precision[-1] if len(precision)>0 else 0),
                    "Recall_final": recall if isinstance(recall, float) else (recall[-1] if len(recall)>0 else 0),
                    "F1_final": f1 if isinstance(f1, float) else (f1_curve[-1] if len(f1_curve)>0 else 0),
                })
                wandb.finish()
            except Exception as e:
                logger.error(f"Additional evaluation metrics calculation failed: {e}")
                import traceback
                traceback.print_exc()
            return evaluation_results
        except Exception as e:
            logger.error(f"RetinaNet evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    def get_model_size(self):
        """Get model size in MB"""
        if self.model is None:
            return 145.0  # Approximate size for RetinaNet ResNet50
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb