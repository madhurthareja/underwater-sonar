import os
import cv2
import numpy as np
import yaml
import hashlib
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLeakageDetector:
    def __init__(self, data_yaml_path):
        self.data_yaml_path = data_yaml_path
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
    
    def detect_all_leakage_types(self):
        """Run comprehensive data leakage detection"""
        logger.info("Starting comprehensive data leakage analysis...")
        
        leakage_report = {
            'identical_images': [],
            'similar_images': [],
            'filename_patterns': [],
            'identical_annotations': [],
            'directory_structure_issues': [],
            'summary': {}
        }
        
        # 1. Check for identical images by hash
        self.check_identical_images(leakage_report)
        
        # 2. Check for similar images by visual features
        self.check_similar_images(leakage_report)
        
        # 3. Check filename patterns (potential video sequences)
        self.check_filename_patterns(leakage_report)
        
        # 4. Check for identical annotations
        self.check_identical_annotations(leakage_report)
        
        # 5. Check directory structure
        self.check_directory_structure(leakage_report)
        
        # Generate summary
        self.generate_summary(leakage_report)
        
        return leakage_report
    
    def check_identical_images(self, report):
        """Check for pixel-perfect identical images across splits"""
        logger.info("Checking for identical images...")
        
        image_hashes = {}
        
        for split in ['train', 'val', 'test']:
            if split not in self.data_config:
                continue
                
            split_dir = Path(self.data_config[split])
            if not split_dir.exists():
                logger.warning(f"Directory not found: {split_dir}")
                continue
            
            # Try different possible image locations
            possible_img_dirs = [
                split_dir,  # Images directly in split directory
                split_dir / 'images',  # Standard YOLO structure
                split_dir / 'img',  # Alternative structure
            ]
            
            img_files = []
            for img_dir in possible_img_dirs:
                if img_dir.exists():
                    logger.info(f"Checking directory: {img_dir}")
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                        found_files = list(img_dir.glob(f'*{ext}'))
                        img_files.extend(found_files)
                        # Also check subdirectories
                        found_files = list(img_dir.glob(f'**/*{ext}'))
                        img_files.extend(found_files)
            
            # Remove duplicates
            img_files = list(set(img_files))
            
            logger.info(f"Found {len(img_files)} images in {split} split at {split_dir}")
            
            # If still no images found, list what's actually in the directory
            if len(img_files) == 0:
                logger.warning(f"No images found in {split_dir}")
                if split_dir.exists():
                    contents = list(split_dir.iterdir())
                    logger.info(f"Directory contents: {[str(p.name) for p in contents[:10]]}")
            
            for img_path in img_files:
                img_hash = self._calculate_image_hash(img_path)
                if img_hash is None:
                    continue
                
                if img_hash in image_hashes:
                    # Found duplicate!
                    report['identical_images'].append({
                        'image1': str(image_hashes[img_hash]['path']),
                        'image2': str(img_path),
                        'split1': image_hashes[img_hash]['split'],
                        'split2': split,
                        'hash': img_hash
                    })
                else:
                    image_hashes[img_hash] = {
                        'path': img_path,
                        'split': split
                    }
    
    def check_similar_images(self, report, similarity_threshold=0.95):
        """Check for visually similar images using histogram comparison"""
        logger.info("Checking for visually similar images...")
        
        split_histograms = {}
        
        # Calculate histograms for each split
        for split in ['train', 'val', 'test']:
            if split not in self.data_config:
                continue
                
            split_dir = Path(self.data_config[split])
            if not split_dir.exists():
                continue
            
            img_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_files.extend(split_dir.glob(f'*{ext}'))
                img_files.extend(split_dir.glob(f'*{ext.upper()}'))
            
            split_histograms[split] = []
            
            # Limit to first 200 images for performance
            for img_path in img_files[:200]:
                hist = self._calculate_histogram(img_path)
                if hist is not None:
                    split_histograms[split].append({
                        'path': str(img_path),
                        'hist': hist
                    })
        
        # Compare across different splits
        split_names = list(split_histograms.keys())
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                logger.info(f"Comparing {split1} vs {split2}...")
                
                for img1 in split_histograms[split1]:
                    for img2 in split_histograms[split2]:
                        similarity = cv2.compareHist(
                            img1['hist'], img2['hist'], cv2.HISTCMP_CORREL
                        )
                        
                        if similarity > similarity_threshold:
                            report['similar_images'].append({
                                'image1': img1['path'],
                                'image2': img2['path'],
                                'split1': split1,
                                'split2': split2,
                                'similarity': similarity
                            })
    
    def check_filename_patterns(self, report):
        """Check for sequential filename patterns that suggest video frames"""
        logger.info("Checking filename patterns...")
        
        all_filenames = {}
        
        for split in ['train', 'val', 'test']:
            if split not in self.data_config:
                continue
                
            split_dir = Path(self.data_config[split])
            if not split_dir.exists():
                continue
            
            img_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_files.extend(split_dir.glob(f'*{ext}'))
                img_files.extend(split_dir.glob(f'*{ext.upper()}'))
            
            all_filenames[split] = [img.stem for img in img_files]
        
        # Look for similar base names across splits
        base_patterns = defaultdict(list)
        
        for split, filenames in all_filenames.items():
            for filename in filenames:
                # Remove numbers and common suffixes to find base pattern
                base_name = self._extract_base_pattern(filename)
                base_patterns[base_name].append({
                    'filename': filename,
                    'split': split
                })
        
        # Find patterns that appear in multiple splits
        for base_pattern, instances in base_patterns.items():
            splits_involved = set(inst['split'] for inst in instances)
            if len(splits_involved) > 1:
                report['filename_patterns'].append({
                    'base_pattern': base_pattern,
                    'instances': instances,
                    'splits_involved': list(splits_involved)
                })
    
    def check_identical_annotations(self, report):
        """Check for identical annotation files"""
        logger.info("Checking for identical annotations...")
        
        annotation_hashes = {}
        
        for split in ['train', 'val', 'test']:
            if split not in self.data_config:
                continue
                
            split_dir = Path(self.data_config[split])
            
            # Try different possible label locations
            possible_label_dirs = [
                split_dir / 'labels',  # Standard YOLO structure
                split_dir.parent / 'labels',  # Labels at same level as images
                split_dir / '../labels',  # Alternative path
                split_dir,  # Labels in same directory as images
            ]
            
            label_files = []
            for label_dir in possible_label_dirs:
                try:
                    label_dir = label_dir.resolve()
                    if label_dir.exists():
                        logger.info(f"Checking label directory: {label_dir}")
                        found_labels = list(label_dir.glob('*.txt'))
                        label_files.extend(found_labels)
                        # Also check subdirectories
                        found_labels = list(label_dir.glob('**/*.txt'))
                        label_files.extend(found_labels)
                except Exception as e:
                    logger.warning(f"Error accessing {label_dir}: {e}")
            
            # Remove duplicates
            label_files = list(set(label_files))
            
            logger.info(f"Found {len(label_files)} label files in {split} split")
            
            if len(label_files) == 0:
                logger.warning(f"No label files found for {split} split")
                # List what's in the split directory
                if split_dir.exists():
                    contents = list(split_dir.iterdir())
                    logger.info(f"Split directory contents: {[str(p.name) for p in contents[:10]]}")
            
            for label_path in label_files:
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    
                    if content:  # Only check non-empty files
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        
                        if content_hash in annotation_hashes:
                            report['identical_annotations'].append({
                                'annotation1': str(annotation_hashes[content_hash]['path']),
                                'annotation2': str(label_path),
                                'split1': annotation_hashes[content_hash]['split'],
                                'split2': split,
                                'content': content[:100] + '...' if len(content) > 100 else content
                            })
                        else:
                            annotation_hashes[content_hash] = {
                                'path': label_path,
                                'split': split
                            }
                except Exception as e:
                    logger.warning(f"Error reading {label_path}: {e}")
    
    def check_directory_structure(self, report):
        """Check for directory structure issues"""
        logger.info("Checking directory structure...")
        
        paths = {}
        for split in ['train', 'val', 'test']:
            if split in self.data_config:
                paths[split] = Path(self.data_config[split]).resolve()
        
        # Check for overlapping or nested directories
        split_names = list(paths.keys())
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                path1, path2 = paths[split1], paths[split2]
                
                if path1 == path2:
                    report['directory_structure_issues'].append({
                        'issue': 'identical_paths',
                        'split1': split1,
                        'split2': split2,
                        'path': str(path1)
                    })
                elif path1 in path2.parents:
                    report['directory_structure_issues'].append({
                        'issue': 'nested_directories',
                        'parent_split': split1,
                        'child_split': split2,
                        'parent_path': str(path1),
                        'child_path': str(path2)
                    })
                elif path2 in path1.parents:
                    report['directory_structure_issues'].append({
                        'issue': 'nested_directories',
                        'parent_split': split2,
                        'child_split': split1,
                        'parent_path': str(path2),
                        'child_path': str(path1)
                    })
    
    def generate_summary(self, report):
        """Generate summary statistics"""
        report['summary'] = {
            'total_identical_images': len(report['identical_images']),
            'total_similar_images': len(report['similar_images']),
            'total_filename_patterns': len(report['filename_patterns']),
            'total_identical_annotations': len(report['identical_annotations']),
            'total_directory_issues': len(report['directory_structure_issues']),
            'leakage_detected': any([
                report['identical_images'],
                report['similar_images'],
                report['filename_patterns'],
                report['identical_annotations'],
                report['directory_structure_issues']
            ])
        }
    
    def _calculate_image_hash(self, img_path):
        """Calculate MD5 hash of image content"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            return hashlib.md5(img.tobytes()).hexdigest()
        except Exception:
            return None
    
    def _calculate_histogram(self, img_path):
        """Calculate color histogram for image comparison"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # Calculate histogram for each channel
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # Normalize histograms
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            
            return np.concatenate([hist_b, hist_g, hist_r])
        except Exception:
            return None
    
    def _extract_base_pattern(self, filename):
        """Extract base pattern from filename by removing numbers and common suffixes"""
        import re
        # Remove common suffixes and numbers
        base = re.sub(r'_?\d+$', '', filename)  # Remove trailing numbers
        base = re.sub(r'_?(frame|img|image)_?\d*', '', base, flags=re.IGNORECASE)
        base = re.sub(r'_+', '_', base)  # Collapse multiple underscores
        return base.strip('_')
    
    def save_report(self, report, save_path='data_leakage_report.txt'):
        """Save detailed leakage report"""
        with open(save_path, 'w') as f:
            f.write("COMPREHENSIVE DATA LEAKAGE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            f.write("SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for key, value in report['summary'].items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n🚨 LEAKAGE DETECTED: {'YES' if report['summary']['leakage_detected'] else 'NO'}\n\n")
            
            # Detailed findings
            sections = [
                ('IDENTICAL IMAGES', 'identical_images'),
                ('SIMILAR IMAGES', 'similar_images'),
                ('FILENAME PATTERNS', 'filename_patterns'),
                ('IDENTICAL ANNOTATIONS', 'identical_annotations'),
                ('DIRECTORY ISSUES', 'directory_structure_issues')
            ]
            
            for section_name, section_key in sections:
                f.write(f"{section_name}:\n")
                f.write("-" * len(section_name) + "\n")
                
                items = report[section_key]
                if not items:
                    f.write("None found.\n\n")
                    continue
                
                for i, item in enumerate(items[:10], 1):  # Show first 10
                    f.write(f"{i}. {item}\n")
                
                if len(items) > 10:
                    f.write(f"... and {len(items) - 10} more\n")
                f.write("\n")
        
        logger.info(f"Detailed report saved to: {save_path}")

def main():
    """Run data leakage detection"""
    data_yaml_path = "/home/madhurthareja/underwater-sonar/dataset/data.yaml"
    
    if not os.path.exists(data_yaml_path):
        logger.error(f"data.yaml not found at: {data_yaml_path}")
        return
    
    detector = DataLeakageDetector(data_yaml_path)
    report = detector.detect_all_leakage_types()
    
    # Print summary
    print("\n" + "="*60)
    print("DATA LEAKAGE ANALYSIS SUMMARY")
    print("="*60)
    
    summary = report['summary']
    print(f"📊 Identical Images: {summary['total_identical_images']}")
    print(f"📊 Similar Images: {summary['total_similar_images']}")
    print(f"📊 Filename Patterns: {summary['total_filename_patterns']}")
    print(f"📊 Identical Annotations: {summary['total_identical_annotations']}")
    print(f"📊 Directory Issues: {summary['total_directory_issues']}")
    print(f"\n🚨 LEAKAGE DETECTED: {'YES - CHECK REPORT!' if summary['leakage_detected'] else 'NO'}")
    
    if summary['leakage_detected']:
        print("\n⚠️  CRITICAL ISSUES FOUND!")
        print("Your perfect scores are likely due to data leakage.")
        print("Check the detailed report for specific issues.")
    
    # Save detailed report
    detector.save_report(report)
    
    return report

if __name__ == "__main__":
    main()