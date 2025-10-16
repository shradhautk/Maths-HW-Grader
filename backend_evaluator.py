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


# ==========================================
# CLI Usage Example
# ==========================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        # CLI usage: python backend_evaluator.py ground_truth.txt prediction.txt
        evaluator = BackendOCREvaluator()
        result = evaluator.evaluate_from_files(sys.argv[1], sys.argv[2])
        
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Demo usage
        evaluator = BackendOCREvaluator()
        
        gt_text = "H0: μ = 182\nHa: μ < 182\nα = 0.05\nz = -2.00\nWe reject H0"
        pred_text = "H0: μ = 182\nHa: μ < 182\nalpha = 0.05\nz = -2.0\nWe reject H0"
        
        result = evaluator.evaluate_from_text(gt_text, pred_text)
        print(json.dumps(result.to_dict(), indent=2))