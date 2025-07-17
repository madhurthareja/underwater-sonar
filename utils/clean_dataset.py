import shutil
from pathlib import Path
import yaml
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCleaner:
    def __init__(self, data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
    
    def create_clean_dataset(self, output_dir, train_ratio=0.7, val_ratio=0.2):
        """Create a clean dataset without leakage"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Collect all unique images
        unique_images = self.collect_unique_images()
        logger.info(f"Found {len(unique_images)} unique images")
        
        # 2. Remove images with identical annotations across splits
        clean_images = self.remove_annotation_duplicates(unique_images)
        logger.info(f"After removing annotation duplicates: {len(clean_images)} images")
        
        # 3. Split cleanly
        self.split_clean_dataset(clean_images, output_dir, train_ratio, val_ratio)
        
        # 4. Create new data.yaml
        self.create_clean_data_yaml(output_dir)
    
    def collect_unique_images(self):
        """Collect all unique images (remove pixel duplicates)"""
        unique_images = {}
        image_hashes = {}
        
        for split in ['train', 'val', 'test']:
            if split not in self.data_config:
                continue
                
            split_dir = Path(self.data_config[split])
            img_dir = split_dir / 'images'
            label_dir = split_dir / 'labels'
            
            if not img_dir.exists():
                continue
            
            for img_path in img_dir.glob('*.bmp'):
                img_hash = self.calculate_image_hash(img_path)
                if img_hash is None:
                    continue
                
                label_path = label_dir / f"{img_path.stem}.txt"
                
                if img_hash not in image_hashes:
                    # First occurrence of this image
                    image_hashes[img_hash] = {
                        'img_path': img_path,
                        'label_path': label_path if label_path.exists() else None,
                        'split': split
                    }
                    unique_images[str(img_path)] = image_hashes[img_hash]
                else:
                    logger.info(f"Skipping duplicate: {img_path} (duplicate of {image_hashes[img_hash]['img_path']})")
        
        return unique_images
    
    def remove_annotation_duplicates(self, unique_images):
        """Remove images with identical annotations"""
        annotation_hashes = {}
        clean_images = {}
        
        for img_key, img_data in unique_images.items():
            label_path = img_data['label_path']
            if label_path is None or not label_path.exists():
                clean_images[img_key] = img_data
                continue
            
            with open(label_path, 'r') as f:
                content = f.read().strip()
            
            if not content:
                clean_images[img_key] = img_data
                continue
            
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in annotation_hashes:
                annotation_hashes[content_hash] = img_data
                clean_images[img_key] = img_data
            else:
                logger.info(f"Skipping image with duplicate annotation: {img_data['img_path']}")
        
        return clean_images
    
    def split_clean_dataset(self, clean_images, output_dir, train_ratio, val_ratio):
        """Split clean images into train/val/test"""
        import random
        
        images_list = list(clean_images.values())
        random.shuffle(images_list)
        
        n_total = len(images_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': images_list[:n_train],
            'val': images_list[n_train:n_train + n_val],
            'test': images_list[n_train + n_val:]
        }
        
        # Create directories and copy files
        for split, images in splits.items():
            img_dir = output_dir / split / 'images'
            label_dir = output_dir / split / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Copying {len(images)} images to {split} split...")
            
            for img_data in images:
                # Copy image
                dst_img = img_dir / img_data['img_path'].name
                shutil.copy2(img_data['img_path'], dst_img)
                
                # Copy label if exists
                if img_data['label_path'] and img_data['label_path'].exists():
                    dst_label = label_dir / img_data['label_path'].name
                    shutil.copy2(img_data['label_path'], dst_label)
        
        logger.info("Clean dataset created successfully!")
        
        # Print statistics
        for split, images in splits.items():
            logger.info(f"{split}: {len(images)} images")
    
    def create_clean_data_yaml(self, output_dir):
        """Create data.yaml for clean dataset"""
        config = {
            'train': str(output_dir / 'train' / 'images'),
            'val': str(output_dir / 'val' / 'images'),
            'test': str(output_dir / 'test' / 'images'),
            'nc': self.data_config['nc'],
            'names': self.data_config['names']
        }
        
        yaml_path = output_dir / 'data_clean.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Clean data.yaml created: {yaml_path}")
        return yaml_path
    
    def calculate_image_hash(self, img_path):
        """Calculate image hash"""
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            return hashlib.md5(img.tobytes()).hexdigest()
        except:
            return None

def main():
    # Clean the dataset
    cleaner = DatasetCleaner('/home/madhurthareja/underwater-sonar/dataset/data.yaml')
    cleaner.create_clean_dataset('/home/madhurthareja/underwater-sonar/dataset_clean')

if __name__ == "__main__":
    main()