#!/usr/bin/env python3
"""
Generate Final Comparison Report from Existing Results
"""

import os
import sys
from comparison_framework import ModelComparisonFramework

def main():
    # Set up paths
    data_yaml_path = '/home/madhurthareja/underwater-sonar/dataset_clean/data_clean.yaml'
    results_dir = '/home/madhurthareja/underwater-sonar/model_comparison/results'
    
    # Initialize comparison framework
    framework = ModelComparisonFramework(data_yaml_path, results_dir)
    
    print("Generating comprehensive comparison report from existing results...")
    print("=" * 60)
    
    # Generate report from existing JSON files
    df = framework.generate_report_from_existing_results()
    
    if df is not None:
        print("\nCOMPARISON RESULTS:")
        print("-" * 40)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Find best performing model
        best_idx = df['mAP@0.5'].idxmax()
        best_model = df.loc[best_idx]
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
        print(f"   mAP@0.5: {best_model['mAP@0.5']:.4f}")
        print(f"   F1-Score: {best_model['F1-Score']:.4f}")
        if best_model['Model_Size_MB'] > 0:
            efficiency = best_model['mAP@0.5'] / best_model['Model_Size_MB']
            print(f"   Efficiency: {efficiency:.4f} mAP/MB")
        
        # Save all results
        framework.save_results_json()
        
        print(f"\nReports generated in: {results_dir}")
        print("- model_comparison.csv")
        print("- model_comparison_plots.png") 
        print("- detailed_comparison_report.txt")
        print("- all_results.json")
        
    else:
        print(" Failed to generate comparison report")
        print("Make sure you have JSON result files in the results directory:")
        print("- aquayolo_results.json")
        print("- frcnn_results.json")
        print("- retinanet_results.json") 
        print("- yolov8n_results.json")

if __name__ == "__main__":
    main()
