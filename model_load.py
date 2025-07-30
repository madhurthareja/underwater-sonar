import os

LABEL_ROOT = '/home/madhurthareja/underwater-sonar/dataset_clean'
MAX_CLASS_INDEX = 10  # Since your `nc: 11`, valid labels are from 0 to 10

def check_labels(subset):
    label_dir = os.path.join(LABEL_ROOT, subset, 'labels')
    invalid_labels = []

    for file in os.listdir(label_dir):
        if not file.endswith('.txt'):
            continue
        path = os.path.join(label_dir, file)
        with open(path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts or not parts[0].isdigit():
                    continue
                class_id = int(parts[0])
                if not (0 <= class_id <= MAX_CLASS_INDEX):
                    invalid_labels.append((path, line_no, class_id, line.strip()))
    
    return invalid_labels

for subset in ['train', 'val', 'test']:
    invalid = check_labels(subset)
    if invalid:
        print(f"\n🚨 Invalid class indices in {subset} set:")
        for file, line_no, class_id, content in invalid:
            print(f" - {file} (line {line_no}): class_id={class_id}, content=`{content}`")
    else:
        print(f"✅ No invalid class IDs in {subset} set.")

