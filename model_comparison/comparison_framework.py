#!/usr/bin/env python3
"""
Multi-Model Comparison Framework for UATD Sonar Dataset
Comparing YOLOv8, FRCNN, RetinaNet, and AquaYOLO
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparisonFramework:
    def __init__(self, data_yaml_path, results_dir='results'):
        self.data_yaml_path = data_yaml_path
        self.results_dir = results_dir
        self.models = {
            'YOLOv8n': None,
            'FRCNN': None, 
            'RetinaNet': None,
            'AquaYOLO': None
        }
        self.results = {}
        os.makedirs(self.results_dir, exist_ok=True)
        
    def register_model(self, model_name, model_instance):
        """Register a model for comparison"""
        if model_name in self.models:
            self.models[model_name] = model_instance
            logger.info(f"Registered {model_name}")
        else:
            logger.warning(f"Unknown model: {model_name}")
    
    def run_evaluation(self, model_name):
        """Run evaluation for a specific model"""
        model = self.models[model_name]
        if model is None:
            logger.error(f"Model {model_name} not registered")
            return None
            
        logger.info(f"Evaluating {model_name}...")
        try:
            # Pass data_yaml to models that need it
            if hasattr(model, 'evaluate'):
                if model_name in ['FRCNN', 'RetinaNet', 'AquaYOLO']:
                    results = model.evaluate(data_yaml=self.data_yaml_path)
                else:
                    results = model.evaluate()
            else:
                logger.error(f"Model {model_name} has no evaluate method")
                return None
                
            self.results[model_name] = results
            
            # Save individual results
            if results:
                result_file = os.path.join(self.results_dir, f'{model_name.lower()}_results.json')
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            return results
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return None
    
    def evaluate_single_model(self, model_name):
        """Evaluate a single model and return results"""
        return self.run_evaluation(model_name)
    
    def compare_all_models(self):
        """Run comparison across all registered models"""
        for model_name, model in self.models.items():
            if model is not None:
                self.run_evaluation(model_name)
        
        return self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            logger.error("No results available for comparison")
            return None
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in self.results.items():
            if results is not None:
                comparison_data.append({
                    'Model': model_name,
                    'mAP@0.5': results.get('map50', 0),
                    'mAP@0.5:0.95': results.get('map', 0),
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1-Score': results.get('f1', 0),
                    'Inference_Time_ms': results.get('inference_time', 0),
                    'Model_Size_MB': results.get('model_size', 0)
                })
            else:
                logger.warning(f"Skipping {model_name} - evaluation failed or returned None")
        
        if not comparison_data:
            logger.error("No valid results to compare")
            return None
            
        df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        csv_path = os.path.join(self.results_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Generate visualizations
        self.create_comparison_plots(df)
        
        # Generate detailed report
        self.create_detailed_report(df)
        
        logger.info(f"Comparison report saved to {self.results_dir}")
        return df
    
    def create_comparison_plots(self, df):
        """Create comparison visualizations"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Performance Comparison - UATD Sonar Dataset', fontsize=16, fontweight='bold')
        
        # 1. mAP Comparison
        ax1 = axes[0]
        models = df['Model']
        map50 = df['mAP@0.5']
        map_range = df['mAP@0.5:0.95']
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, map50, width, label='mAP@0.5', alpha=0.8, color='#2E8B57')
        bars2 = ax1.bar(x + width/2, map_range, width, label='mAP@0.5:0.95', alpha=0.8, color='#4169E1')
        
        ax1.set_ylabel('mAP Score', fontweight='bold')
        ax1.set_title('Mean Average Precision Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. F1-Score Comparison
        ax2 = axes[1]
        bars = ax2.bar(models, df['F1-Score'], alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_ylabel('F1-Score', fontweight='bold')
        ax2.set_title('F1-Score Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, df['F1-Score']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'model_comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plots saved to {plot_path}")
        
        plt.show()
    
    def create_detailed_report(self, df):
        """Generate detailed text report"""
        report_path = os.path.join(self.results_dir, 'detailed_comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_yaml_path}\n")
            f.write(f"Models Compared: {len(df)} models\n\n")
            
            # Performance Summary Table
            f.write("PERFORMANCE METRICS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n")
            
            # Ranking Analysis
            f.write("RANKING ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
            for metric in metrics:
                if metric in df.columns:
                    best_idx = df[metric].idxmax()
                    best_model = df.loc[best_idx, 'Model']
                    best_score = df.loc[best_idx, metric]
                    f.write(f"Best {metric}: {best_model} ({best_score:.4f})\n")
            
            f.write("\n")
            
            # Overall best model
            overall_best_idx = df['mAP@0.5'].idxmax()
            overall_best = df.loc[overall_best_idx]
            f.write(f"OVERALL BEST PERFORMING MODEL:\n")
            f.write(f"Model: {overall_best['Model']}\n")
            f.write(f"mAP@0.5: {overall_best['mAP@0.5']:.4f}\n")
            f.write(f"mAP@0.5:0.95: {overall_best['mAP@0.5:0.95']:.4f}\n")
            f.write(f"Precision: {overall_best['Precision']:.4f}\n")
            f.write(f"Recall: {overall_best['Recall']:.4f}\n")
            f.write(f"F1-Score: {overall_best['F1-Score']:.4f}\n")
            
            if overall_best['Model_Size_MB'] > 0:
                f.write(f"Model Size: {overall_best['Model_Size_MB']:.1f} MB\n")
            
            f.write("\n")
            
            # Model-specific analysis
            f.write("MODEL-SPECIFIC ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            
            for _, row in df.iterrows():
                f.write(f"\n{row['Model']}:\n")
                f.write(f"  - Strong in: ")
                strengths = []
                if row['Precision'] >= df['Precision'].mean():
                    strengths.append("Precision")
                if row['Recall'] >= df['Recall'].mean():
                    strengths.append("Recall")
                if row['mAP@0.5'] >= df['mAP@0.5'].mean():
                    strengths.append("mAP@0.5")
                
                f.write(", ".join(strengths) if strengths else "None identified")
                f.write("\n")
                
                f.write(f"  - Performance Score: {row['F1-Score']:.4f}\n")
                if row['Model_Size_MB'] > 0:
                    f.write(f"  - Efficiency: {row['mAP@0.5']/row['Model_Size_MB']:.4f} mAP/MB\n")
        
        logger.info(f"Detailed report saved to {report_path}")
    
    def load_existing_results(self):
        """Load existing results from JSON files"""
        model_files = {
            'AQUAYOLO': 'aquayolo_results.json',
            'FRCNN': 'frcnn_results.json', 
            'RETINANET': 'retinanet_results.json',
            'YOLOV8N': 'yolov8n_results.json'
        }
        
        for model_name, filename in model_files.items():
            json_path = os.path.join(self.results_dir, filename)
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle nested structure: if data has 'results' key, use that
                    if 'results' in data:
                        self.results[model_name] = data['results']
                    else:
                        self.results[model_name] = data
                    
                    logger.info(f"Loaded existing results for {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
            else:
                logger.warning(f"Results file not found: {json_path}")
        
        return len(self.results) > 0
    
    def generate_report_from_existing_results(self):
        """Generate comparison report from existing JSON files"""
        if self.load_existing_results():
            return self.generate_comparison_report()
        else:
            logger.error("No existing results found to generate report")
            return None

    def save_results_json(self):
        """Save all results to a JSON file"""
        json_path = os.path.join(self.results_dir, 'all_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"All results saved to {json_path}")
