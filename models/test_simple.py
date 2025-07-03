#!/usr/bin/env python3
import os
import sys

# Set environment variables to prevent hanging
os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['YOLO_VERBOSE'] = 'False'

print("Starting simple training script...")

try:
    print("Importing torch...")
    import torch
    print(f"✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    
    print("Importing ultralytics...")
    from ultralytics import YOLO
    print("✓ Ultralytics imported")
    
    print("Creating model...")
    # Try with YOLOv8 first (more stable)
    model = YOLO('yolov8n.pt')
    print("✓ Model created")
    
    print("Starting training...")
    results = model.train(
        data='dataset/data.yaml',
        epochs=5,  # Short test
        batch=2,   # Small batch
        imgsz=320, # Smaller image size
        device=0,
        project='test_runs',
        name='simple_test',
        verbose=True,
        cache=False,
        workers=0  # Single-threaded
    )
    
    print("✓ Training completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()