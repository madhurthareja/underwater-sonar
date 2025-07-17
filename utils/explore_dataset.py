import os
from pathlib import Path
import yaml

def explore_dataset_structure(data_yaml_path):
    """Explore the actual dataset structure"""
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("DATASET STRUCTURE EXPLORATION")
    print("=" * 50)
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
        
        split_path = Path(data_config[split])
        print(f"\n{split.upper()} SPLIT: {split_path}")
        print("-" * 40)
        
        if not split_path.exists():
            print(f"❌ Path does not exist!")
            continue
        
        print(f"✅ Path exists")
        
        # List top-level contents
        try:
            contents = list(split_path.iterdir())
            print(f"📁 Contains {len(contents)} items:")
            
            for item in contents[:20]:  # Show first 20 items
                if item.is_dir():
                    subcontents = list(item.iterdir())
                    print(f"  📁 {item.name}/ ({len(subcontents)} items)")
                    
                    # If it's an images or labels directory, show some files
                    if item.name in ['images', 'labels', 'img']:
                        for subitem in subcontents[:5]:
                            print(f"    📄 {subitem.name}")
                        if len(subcontents) > 5:
                            print(f"    ... and {len(subcontents) - 5} more")
                else:
                    print(f"  📄 {item.name}")
            
            if len(contents) > 20:
                print(f"  ... and {len(contents) - 20} more items")
            
            # Count image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_count = 0
            label_count = 0
            
            for ext in image_extensions:
                image_count += len(list(split_path.glob(f'**/*{ext}')))
                image_count += len(list(split_path.glob(f'**/*{ext.upper()}')))
            
            label_count = len(list(split_path.glob('**/*.txt')))
            
            print(f"📊 Total images found (recursive): {image_count}")
            print(f"📊 Total .txt files found (recursive): {label_count}")
            
        except PermissionError:
            print(f"❌ Permission denied accessing {split_path}")
        except Exception as e:
            print(f"❌ Error exploring {split_path}: {e}")

if __name__ == "__main__":
    data_yaml_path = "/home/madhurthareja/underwater-sonar/dataset/data.yaml"
    explore_dataset_structure(data_yaml_path)