#!/usr/bin/env python3
"""
Safe run complete model comparison
Handles memory issues and runs models sequentially
"""

import sys
import os
import gc
import torch
sys.path.append('/home/madhurthareja/underwater-sonar')
sys.path.append('/home/madhurthareja/underwater-sonar/model_comparison')

from comparison_framework import ModelComparisonFramework
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_model(model_name, data_yaml, results_dir, yolo_weights):
    """Run a single model evaluation safely"""
    try:
        # Clear any existing memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        framework = ModelComparisonFramework(data_yaml, results_dir)
        
        if model_name == 'yolov8':
            from yolov8.yolov8_wrapper import YOLOv8Wrapper
            model = YOLOv8Wrapper(model_path=yolo_weights, data_yaml=data_yaml)
            framework.register_model('YOLOv8n', model)
        elif model_name == 'aquayolo':
            from aquayolo.aquayolo_model import AquaYOLOModel
            model = AquaYOLOModel(nc=10)
            model.create_aqua_config()
            framework.register_model('AquaYOLO', model)
        elif model_name == 'frcnn':
            from frcnn.frcnn_model import FRCNNModel
            model = FRCNNModel(num_classes=12)  # 11 classes + background
            framework.register_model('FRCNN', model)
        elif model_name == 'retinanet':
            from retinanet.retinanet_model import RetinaNetModel
            model = RetinaNetModel(num_classes=12)  # 11 classes (no background)
            framework.register_model('RetinaNet', model)
        
        # Evaluate single model
        model_key = 'YOLOv8n' if model_name == 'yolov8' else model_name.upper()
        if model_name == 'aquayolo':
            model_key = 'AquaYOLO'
        elif model_name == 'frcnn':
            model_key = 'FRCNN'
        elif model_name == 'retinanet':
            model_key = 'RetinaNet'
            
        results = framework.evaluate_single_model(model_key)
        
        # Save individual results
        if results:
            import json
            results_file = os.path.join(results_dir, f'{model_name}_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'model': model_name,
                    'results': results
                }, f, indent=2)
            logger.info(f"Saved {model_name} results to {results_file}")
            return results
            
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return None
    finally:
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Safely compare multiple object detection models')
    parser.add_argument('--data', type=str, 
                       default='/home/madhurthareja/underwater-sonar/dataset_clean/data_clean.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--yolo-weights', type=str,
                       default='/home/madhurthareja/underwater-sonar/runs_train/uatd_sonar_exp/weights/best.pt',
                       help='Path to trained YOLOv8 weights')
    parser.add_argument('--results-dir', type=str,
                       default='/home/madhurthareja/underwater-sonar/model_comparison/results',
                       help='Directory to save comparison results')
    parser.add_argument('--models', nargs='+', 
                       default=['yolov8', 'frcnn', 'retinanet', 'aquayolo'],
                       choices=['yolov8', 'frcnn', 'retinanet', 'aquayolo'],
                       help='Models to compare')
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("\\n" + "="*60)
    print("UNDERWATER SONAR OBJECT DETECTION MODEL COMPARISON (SAFE)")
    print("="*60)
    print(f"Dataset: {os.path.basename(args.data)}")
    print(f"Models: {', '.join(args.models)}")
    print("="*60 + "\\n")
    
    all_results = {}
    
    # Run each model individually to avoid memory conflicts
    for model_name in args.models:
        logger.info(f"Evaluating {model_name.upper()}...")
        print(f"\\n--- Evaluating {model_name.upper()} ---")
        
        result = run_single_model(model_name, args.data, args.results_dir, args.yolo_weights)
        if result:
            all_results[model_name] = result
            print(f"✓ {model_name.upper()} completed: mAP@0.5 = {result.get('map50', 0):.4f}")
        else:
            print(f"✗ {model_name.upper()} failed")
    
    # Create final comparison
    if all_results:
        print("\\n" + "="*60)
        print("FINAL COMPARISON RESULTS")
        print("="*60)
        
        import pandas as pd
        comparison_data = []
        for model_name, results in all_results.items():
            comparison_data.append({
                'Model': model_name.upper() if model_name != 'yolov8' else 'YOLOv8n',
                'mAP@0.5': results.get('map50', 0),
                'mAP@0.5:0.95': results.get('map', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1', 0),
                'Inference_Time_ms': results.get('inference_time', 0),
                'Model_Size_MB': results.get('model_size', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('mAP@0.5', ascending=False)
        
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Save comparison CSV
        csv_path = os.path.join(args.results_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"\\nResults saved to: {csv_path}")
        
        # Save all results JSON
        json_path = os.path.join(args.results_dir, 'all_results.json')
        import json
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to: {json_path}")
        
        # Highlight best model
        best_model = df.iloc[0]
        print(f"\\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
        print(f"   mAP@0.5: {best_model['mAP@0.5']:.4f}")
        print(f"   F1-Score: {best_model['F1-Score']:.4f}")
        if best_model['Model_Size_MB'] > 0:
            efficiency = best_model['mAP@0.5'] / best_model['Model_Size_MB']
            print(f"   Efficiency: {efficiency:.4f} mAP/MB")
    
    return 0

if __name__ == "__main__":
    exit(main())
