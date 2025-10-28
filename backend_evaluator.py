"""
Backend-ready OCR Evaluator for Statistics Exams
Optimized for API integration with FastAPI/Flask
"""

import re
import json
from typing import Dict, List, Optional, Union
from difflib import SequenceMatcher
import Levenshtein
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EvaluationResult:
    """Structured evaluation result for API responses"""
    # Character-level metrics
    character_error_rate: float
    word_error_rate: float
    levenshtein_distance: int
    text_similarity: float
    
    # Notation metrics
    notation_precision: float
    notation_recall: float
    notation_f1: float
    notation_details: Dict
    
    # Numerical metrics
    numbers_matched: int
    numbers_total: int
    numerical_accuracy: float
    
    # Semantic metrics
    avg_sentence_similarity: float
    sentences_matched_80: int
    sentences_total: int
    
    # Summary
    overall_quality: str  # "excellent", "good", "fair", "poor"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class BackendOCREvaluator:
    """
    Production-ready evaluator optimized for backend integration
    - Handles files, strings, and byte streams
    - Returns structured results
    - Includes error handling
    - Optimized for performance
    """
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        
        # Statistical notation patterns
        self.notation_patterns = {
            'H0': r'H[0o]:?\s*[μµπpρ]?\s*[=≤≥<>]',
            'Ha': r'H[a1]:?\s*[μµπpρ]?\s*[=≤≥<>≠!=]',
            'alpha': r'[αa]\s*=\s*0?\.\d+',
            'z_score': r'[zZ]\s*=\s*-?\d+\.?\d*',
            't_score': r'[tT]\s*=\s*-?\d+\.?\d*',
            'p_value': r'[pP][\s-]*value\s*[=:]?\s*0?\.\d+',
            'probability': r'[pP]\([^)]+\)\s*=\s*0?\.\d+',
            'mean': r'[μxX̄]\s*=\s*\d+\.?\d*',
            'std_dev': r'[σs]\s*=\s*\d+\.?\d*',
            'correlation': r'[ρr]\s*=\s*-?0?\.\d+',
            'confidence_interval': r'\(\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\)',
        }
    
    # ==========================================
    # Main Evaluation Methods (Multiple Inputs)
    # ==========================================
    
    def evaluate_from_files(self, 
                           ground_truth_path: str, 
                           prediction_path: str) -> EvaluationResult:
        """
        Evaluate from file paths
        
        Args:
            ground_truth_path: Path to ground truth text file
            prediction_path: Path to prediction text file
            
        Returns:
            EvaluationResult object
        """
        try:
            gt_text = Path(ground_truth_path).read_text(encoding='utf-8')
            pred_text = Path(prediction_path).read_text(encoding='utf-8')
            return self.evaluate_from_text(gt_text, pred_text)
        except Exception as e:
            raise ValueError(f"Error reading files: {e}")
    
    def evaluate_from_text(self, 
                          ground_truth: str, 
                          prediction: str) -> EvaluationResult:
        """
        Evaluate from text strings (main method)
        
        Args:
            ground_truth: Ground truth text
            prediction: Predicted text
            
        Returns:
            EvaluationResult object
        """
        try:
            # Calculate all metrics
            cer = self._calculate_cer(prediction, ground_truth)
            wer = self._calculate_wer(prediction, ground_truth)
            lev_dist = self._calculate_levenshtein(prediction, ground_truth)
            similarity = self._calculate_similarity(prediction, ground_truth)
            
            notation = self._detect_notation(prediction, ground_truth)
            notation_metrics = self._calculate_notation_metrics(notation)
            
            numbers = self._extract_and_compare_numbers(prediction, ground_truth)
            
            sentences = self._compare_sentences(prediction, ground_truth)
            
            # Determine overall quality
            quality = self._determine_quality(cer, wer, notation_metrics['f1'])
            
            return EvaluationResult(
                character_error_rate=cer,
                word_error_rate=wer,
                levenshtein_distance=lev_dist,
                text_similarity=similarity,
                notation_precision=notation_metrics['precision'],
                notation_recall=notation_metrics['recall'],
                notation_f1=notation_metrics['f1'],
                notation_details=notation,
                numbers_matched=numbers['matched'],
                numbers_total=numbers['gt_count'],
                numerical_accuracy=numbers['accuracy'],
                avg_sentence_similarity=sentences['avg_similarity'],
                sentences_matched_80=sentences['sentences_matched_above_80'],
                sentences_total=sentences['gt_sentence_count'],
                overall_quality=quality
            )
        except Exception as e:
            raise ValueError(f"Evaluation error: {e}")
    
    def evaluate_batch(self, 
                      evaluations: List[Dict[str, str]]) -> List[EvaluationResult]:
        """
        Batch evaluation for multiple file pairs
        
        Args:
            evaluations: List of dicts with 'ground_truth' and 'prediction' keys
            
        Returns:
            List of EvaluationResult objects
            
        Example:
            evaluations = [
                {'ground_truth': 'text1...', 'prediction': 'pred1...'},
                {'ground_truth': 'text2...', 'prediction': 'pred2...'},
            ]
        """
        results = []
        for item in evaluations:
            try:
                result = self.evaluate_from_text(
                    item['ground_truth'], 
                    item['prediction']
                )
                results.append(result)
            except Exception as e:
                # Include error in results
                results.append({
                    'error': str(e),
                    'ground_truth': item.get('ground_truth', '')[:100],
                    'prediction': item.get('prediction', '')[:100]
                })
        return results
    
    # ==========================================
    # Core Metric Calculations (Private)
    # ==========================================
    
    def _calculate_cer(self, pred: str, gt: str) -> float:
        """Character Error Rate"""
        if not gt:
            return 1.0 if pred else 0.0
        distance = Levenshtein.distance(pred, gt)
        return distance / len(gt)
    
    def _calculate_wer(self, pred: str, gt: str) -> float:
        """Word Error Rate - proper word-level alignment"""
        pred_words = pred.split()
        gt_words = gt.split()
        
        if not gt_words:
            return 1.0 if pred_words else 0.0
        
        # Word-level edit distance
        distance = self._word_edit_distance(gt_words, pred_words)
        return distance / len(gt_words)
    
    def _word_edit_distance(self, gt_words: List[str], pred_words: List[str]) -> int:
        """Calculate word-level edit distance using dynamic programming"""
        if not gt_words:
            return len(pred_words)
        if not pred_words:
            return len(gt_words)
        
        d = [[0] * (len(pred_words) + 1) for _ in range(len(gt_words) + 1)]
        
        for i in range(len(gt_words) + 1):
            d[i][0] = i
        for j in range(len(pred_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(gt_words) + 1):
            for j in range(1, len(pred_words) + 1):
                cost = 0 if gt_words[i-1] == pred_words[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,
                    d[i][j-1] + 1,
                    d[i-1][j-1] + cost
                )
        
        return d[len(gt_words)][len(pred_words)]
    
    def _calculate_levenshtein(self, pred: str, gt: str) -> int:
        """Raw Levenshtein distance"""
        return Levenshtein.distance(pred, gt)
    
    def _calculate_similarity(self, pred: str, gt: str) -> float:
        """Text similarity using SequenceMatcher"""
        return SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
    
    def _detect_notation(self, pred: str, gt: str) -> Dict:
        """Detect statistical notation"""
        results = {}
        for notation_name, pattern in self.notation_patterns.items():
            gt_matches = re.findall(pattern, gt, re.IGNORECASE)
            pred_matches = re.findall(pattern, pred, re.IGNORECASE)
            
            results[notation_name] = {
                'in_gt': len(gt_matches) > 0,
                'in_pred': len(pred_matches) > 0,
                'correct': (len(gt_matches) > 0) == (len(pred_matches) > 0)
            }
        return results
    
    def _calculate_notation_metrics(self, notation_results: Dict) -> Dict:
        """Calculate precision, recall, F1"""
        tp = sum(1 for v in notation_results.values() if v['in_gt'] and v['in_pred'])
        fp = sum(1 for v in notation_results.values() if not v['in_gt'] and v['in_pred'])
        fn = sum(1 for v in notation_results.values() if v['in_gt'] and not v['in_pred'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _extract_and_compare_numbers(self, pred: str, gt: str) -> Dict:
        """Extract and compare numbers"""
        number_pattern = r'-?\d+\.?\d*'
        gt_numbers = [float(n) for n in re.findall(number_pattern, gt)]
        pred_numbers = [float(n) for n in re.findall(number_pattern, pred)]
        
        matched = sum(
            1 for gt_num in gt_numbers
            if any(abs(gt_num - pred_num) < self.tolerance for pred_num in pred_numbers)
        )
        
        return {
            'gt_count': len(gt_numbers),
            'pred_count': len(pred_numbers),
            'matched': matched,
            'accuracy': matched / len(gt_numbers) if gt_numbers else 0.0
        }
    
    def _compare_sentences(self, pred: str, gt: str) -> Dict:
        """Compare sentence-by-sentence"""
        gt_sentences = [s.strip() for s in re.split(r'[.!?]+', gt) if s.strip()]
        pred_sentences = [s.strip() for s in re.split(r'[.!?]+', pred) if s.strip()]
        
        similarities = [
            max([SequenceMatcher(None, gt_sent.lower(), pred_sent.lower()).ratio()
                 for pred_sent in pred_sentences], default=0.0)
            for gt_sent in gt_sentences
        ]
        
        return {
            'gt_sentence_count': len(gt_sentences),
            'pred_sentence_count': len(pred_sentences),
            'avg_similarity': sum(similarities) / len(similarities) if similarities else 0.0,
            'sentences_matched_above_80': sum(1 for s in similarities if s > 0.8),
            'sentences_matched_above_60': sum(1 for s in similarities if s > 0.6)
        }
    
    def _determine_quality(self, cer: float, wer: float, notation_f1: float) -> str:
        """Determine overall quality rating"""
        score = (1 - cer) * 0.4 + (1 - wer) * 0.3 + notation_f1 * 0.3
        
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "fair"
        else:
            return "poor"
    
    # ==========================================
    # Utility Methods
    # ==========================================
    
    def get_summary_stats(self, results: List[EvaluationResult]) -> Dict:
        """Calculate aggregate statistics for batch results"""
        if not results:
            return {}
        
        return {
            'total_evaluations': len(results),
            'avg_cer': sum(r.character_error_rate for r in results) / len(results),
            'avg_wer': sum(r.word_error_rate for r in results) / len(results),
            'avg_text_similarity': sum(r.text_similarity for r in results) / len(results),
            'avg_notation_f1': sum(r.notation_f1 for r in results) / len(results),
            'avg_numerical_accuracy': sum(r.numerical_accuracy for r in results) / len(results),
            'quality_distribution': {
                'excellent': sum(1 for r in results if r.overall_quality == 'excellent'),
                'good': sum(1 for r in results if r.overall_quality == 'good'),
                'fair': sum(1 for r in results if r.overall_quality == 'fair'),
                'poor': sum(1 for r in results if r.overall_quality == 'poor'),
            }
        }


# ==========================================
# FastAPI Integration Example
# ==========================================

"""
# Install: pip install fastapi uvicorn python-multipart

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="OCR Evaluation API")
evaluator = BackendOCREvaluator()

class TextEvaluationRequest(BaseModel):
    ground_truth: str
    prediction: str

@app.post("/evaluate/text")
async def evaluate_text(request: TextEvaluationRequest):
    try:
        result = evaluator.evaluate_from_text(
            request.ground_truth, 
            request.prediction
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate/files")
async def evaluate_files(
    ground_truth: UploadFile = File(...),
    prediction: UploadFile = File(...)
):
    try:
        gt_text = (await ground_truth.read()).decode('utf-8')
        pred_text = (await prediction.read()).decode('utf-8')
        
        result = evaluator.evaluate_from_text(gt_text, pred_text)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate/batch")
async def evaluate_batch(evaluations: List[TextEvaluationRequest]):
    try:
        items = [{'ground_truth': e.ground_truth, 'prediction': e.prediction} 
                 for e in evaluations]
        results = evaluator.evaluate_batch(items)
        summary = evaluator.get_summary_stats(results)
        
        return {
            'results': [r.to_dict() if isinstance(r, EvaluationResult) else r 
                       for r in results],
            'summary': summary
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run with: uvicorn backend_evaluator:app --reload
"""


# ==========================================
# Flask Integration Example
# ==========================================

"""
# Install: pip install flask

from flask import Flask, request, jsonify

app = Flask(__name__)
evaluator = BackendOCREvaluator()

@app.route('/evaluate/text', methods=['POST'])
def evaluate_text():
    try:
        data = request.get_json()
        result = evaluator.evaluate_from_text(
            data['ground_truth'], 
            data['prediction']
        )
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/evaluate/files', methods=['POST'])
def evaluate_files():
    try:
        gt_file = request.files['ground_truth']
        pred_file = request.files['prediction']
        
        gt_text = gt_file.read().decode('utf-8')
        pred_text = pred_file.read().decode('utf-8')
        
        result = evaluator.evaluate_from_text(gt_text, pred_text)
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/evaluate/batch', methods=['POST'])
def evaluate_batch():
    try:
        data = request.get_json()
        results = evaluator.evaluate_batch(data['evaluations'])
        summary = evaluator.get_summary_stats(results)
        
        return jsonify({
            'results': [r.to_dict() if isinstance(r, EvaluationResult) else r 
                       for r in results],
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run with: python backend_evaluator.py
if __name__ == '__main__':
    app.run(debug=True)
"""


# # ==========================================
# # CLI Usage Example
# # ==========================================

# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) == 3:
#         # CLI usage: python backend_evaluator.py ground_truth.txt prediction.txt
#         evaluator = BackendOCREvaluator()
#         result = evaluator.evaluate_from_files(sys.argv[1], sys.argv[2])
        
#         print(json.dumps(result.to_dict(), indent=2))
#     else:
#         # Demo usage
#         evaluator = BackendOCREvaluator()
        
#         gt_text = "H0: μ = 182\nHa: μ < 182\nα = 0.05\nz = -2.00\nWe reject H0"
#         pred_text = "H0: μ = 182\nHa: μ < 182\nalpha = 0.05\nz = -2.0\nWe reject H0"
        
#         result = evaluator.evaluate_from_text(gt_text, pred_text)
#         print(json.dumps(result.to_dict(), indent=2))



# ==========================================
# CLI Usage Example
# ==========================================

import os
import csv
import json
from pathlib import Path
from datetime import datetime


def flatten_test_results(test_data, test_id=None):
    """
    Flatten nested JSON test results into a single row dictionary.
    """
    flat_data = {}
    
    # Add test identifier
    if test_id:
        flat_data['test_id'] = test_id
    flat_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Add top-level metrics
    flat_data['character_error_rate'] = test_data.get('character_error_rate')
    flat_data['word_error_rate'] = test_data.get('word_error_rate')
    flat_data['levenshtein_distance'] = test_data.get('levenshtein_distance')
    flat_data['text_similarity'] = test_data.get('text_similarity')
    flat_data['notation_precision'] = test_data.get('notation_precision')
    flat_data['notation_recall'] = test_data.get('notation_recall')
    flat_data['notation_f1'] = test_data.get('notation_f1')
    flat_data['numbers_matched'] = test_data.get('numbers_matched')
    flat_data['numbers_total'] = test_data.get('numbers_total')
    flat_data['numerical_accuracy'] = test_data.get('numerical_accuracy')
    flat_data['avg_sentence_similarity'] = test_data.get('avg_sentence_similarity')
    flat_data['sentences_matched_80'] = test_data.get('sentences_matched_80')
    flat_data['sentences_total'] = test_data.get('sentences_total')
    flat_data['overall_quality'] = test_data.get('overall_quality')
    
    # Add notation details (flatten each notation type)
    notation_details = test_data.get('notation_details', {})
    for notation_type, details in notation_details.items():
        flat_data[f'notation_{notation_type}_in_gt'] = details.get('in_gt')
        flat_data[f'notation_{notation_type}_in_pred'] = details.get('in_pred')
        flat_data[f'notation_{notation_type}_correct'] = details.get('correct')
    
    return flat_data


def save_test_result(test_data, csv_filename, test_id=None):
    """
    Save a single test result to CSV file (appends if file exists).
    """
    flat_data = flatten_test_results(test_data, test_id)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_filename)
    
    # Write to CSV
    with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_data.keys())
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(flat_data)
    
    print(f"✓ Test result saved: {test_id}")


def evaluate_prompt_version(
    prompt_version,
    original_text_path,
    extracted_text_path,
    evaluator,
    file_extension='.txt'
):
    """
    Evaluate all files for a given prompt version.
    
    Args:
        prompt_version: Name of the prompt version (e.g., 'promptv1', 'promptv2')
        original_text_path: Directory containing ground truth text files
        extracted_text_path: Directory containing extracted text files for this prompt version
        evaluator: BackendOCREvaluator instance
        file_extension: File extension to look for (default: '.txt')
    """
    print(f"\n{'='*60}")
    print(f"Starting evaluation for: {prompt_version}")
    print(f"{'='*60}\n")
    
    # Create output CSV filename
    output_csv = f"{prompt_version}.csv"
    
    # Get all files from original text directory
    original_path = Path(original_text_path)
    if not original_path.exists():
        print(f"Error: Original text path '{original_text_path}' does not exist!")
        return
    
    extracted_path = Path(extracted_text_path)
    if not extracted_path.exists():
        print(f"Error: Extracted text path '{extracted_text_path}' does not exist!")
        return
    
    # Get list of all files with specified extension
    original_files = sorted([f for f in original_path.iterdir() 
                            if f.is_file() and f.suffix == file_extension])
    
    if not original_files:
        print(f"Warning: No files with extension '{file_extension}' found in '{original_text_path}'")
        return
    
    print(f"Found {len(original_files)} files to evaluate\n")
    
    # Track statistics
    successful_evaluations = 0
    failed_evaluations = 0
    missing_extracted_files = []
    
    # Evaluate each file
    for idx, original_file in enumerate(original_files, 1):
        file_name = original_file.stem  # Filename without extension
        
        # Construct corresponding extracted file path
        extracted_file = extracted_path / original_file.name
        
        # Check if extracted file exists
        if not extracted_file.exists():
            print(f"⚠ [{idx}/{len(original_files)}] Missing extracted file: {original_file.name}")
            missing_extracted_files.append(original_file.name)
            failed_evaluations += 1
            continue
        
        # Create test ID
        test_id = f"{file_name}_{prompt_version}_test"
        
        try:
            # Perform evaluation
            print(f"[{idx}/{len(original_files)}] Evaluating: {original_file.name}")
            result = evaluator.evaluate_from_files(
                str(original_file),
                str(extracted_file)
            )
            
            # Convert result to dictionary
            result_dict = result.to_dict()
            
            # Save result to CSV
            save_test_result(result_dict, output_csv, test_id)
            successful_evaluations += 1
            
        except Exception as e:
            print(f"✗ [{idx}/{len(original_files)}] Error evaluating {original_file.name}: {str(e)}")
            failed_evaluations += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary for {prompt_version}")
    print(f"{'='*60}")
    print(f"Total files processed: {len(original_files)}")
    print(f"Successful evaluations: {successful_evaluations}")
    print(f"Failed evaluations: {failed_evaluations}")
    
    if missing_extracted_files:
        print(f"\nMissing extracted files ({len(missing_extracted_files)}):")
        for missing_file in missing_extracted_files:
            print(f"  - {missing_file}")
    
    print(f"\nResults saved to: {output_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Import your evaluator class
    # from your_module import BackendOCREvaluator
    
    if len(sys.argv) >= 4:
        # CLI usage with prompt version
        # python script.py <prompt_version> <original_text_dir> <extracted_text_dir>
        
        prompt_version = sys.argv[1]
        original_text_path = sys.argv[2]
        extracted_text_path = sys.argv[3]
        
        # Initialize evaluator
        evaluator = BackendOCREvaluator()
        
        # Run evaluation
        evaluate_prompt_version(
            prompt_version=prompt_version,
            original_text_path=original_text_path,
            extracted_text_path=extracted_text_path,
            evaluator=evaluator
        )
        
    else:
        # Interactive mode - prompt user for inputs
        print("OCR Evaluation Script - Interactive Mode")
        print("=" * 60)
        
        prompt_version = input("Enter prompt version name (e.g., promptv1): ").strip()
        original_text_path = input("Enter path to original/ground truth text files: ").strip()
        extracted_text_path = input("Enter path to extracted text files: ").strip()
        
        if not prompt_version:
            print("Error: Prompt version name cannot be empty!")
            sys.exit(1)
        
        # Initialize evaluator
        evaluator = BackendOCREvaluator()
        
        # Run evaluation
        evaluate_prompt_version(
            prompt_version=prompt_version,
            original_text_path=original_text_path,
            extracted_text_path=extracted_text_path,
            evaluator=evaluator
        )


# Example usage in code:

# Initialize evaluator
evaluator = BackendOCREvaluator()

# Evaluate prompt version 1
evaluate_prompt_version(
    prompt_version='promptv1',
    original_text_path='data/original_texts',
    extracted_text_path='data/promptv1_extracted',
    evaluator=evaluator
)

# # Evaluate prompt version 2
# evaluate_prompt_version(
#     prompt_version='promptv2',
#     original_text_path='data/original_texts',
#     extracted_text_path='data/promptv2_extracted',
#     evaluator=evaluator
# )
# 
