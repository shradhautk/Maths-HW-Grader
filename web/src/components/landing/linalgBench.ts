// LinAlg-Bench figures shown on the Research page.
//
// Source: https://huggingface.co/datasets/LinAlgBench/linalg-bench (CC BY 4.0),
// files results/accuracy/train-0000{0,1,2}.parquet and
// results/error_tags/train-0000{0,1,2}.parquet (3×3, 4×4, 5×5), read 2026-11-14.
// Totals are across all 10 models (220 problems × 10 models = 2,200 responses per size).
//
// The dataset card asks that results not be used for marketing claims or model
// rankings, so the site shows only these all-model totals, never a per-model leaderboard.

export type MatrixSize = '3×3' | '4×4' | '5×5'

export const SIZES: MatrixSize[] = ['3×3', '4×4', '5×5']

export const ACCURACY: Record<MatrixSize, { pass: number; fail: number; total: number; percent: number }> = {
  '3×3': { pass: 2100, fail: 100, total: 2200, percent: 95.5 },
  '4×4': { pass: 1806, fail: 394, total: 2200, percent: 82.1 },
  '5×5': { pass: 1538, fail: 662, total: 2200, percent: 69.9 },
}

/**
 * Classified failures by error type and matrix size. "Other" groups the rarer tags
 * (carry-down, false verification, formatting mismatch, generation loop/truncation,
 * memory loss, variable entanglement, unmapped). Each size's counts add up to its fail total.
 */
export const FAILURE_TYPES: { type: string; meaning: string; counts: Record<MatrixSize, number> }[] = [
  {
    type: 'Hallucination',
    meaning: 'A value or step that does not follow from the earlier working',
    counts: { '3×3': 17, '4×4': 107, '5×5': 312 },
  },
  { type: 'Arithmetic', meaning: 'A calculation slip, such as a wrong product or sum', counts: { '3×3': 29, '4×4': 76, '5×5': 118 } },
  { type: 'Sign error', meaning: 'A plus or minus sign lost or flipped', counts: { '3×3': 33, '4×4': 123, '5×5': 114 } },
  { type: 'Input transcription', meaning: 'A number copied wrongly from the problem', counts: { '3×3': 8, '4×4': 15, '5×5': 47 } },
  { type: 'Method failure', meaning: 'The wrong method, or a correct one applied wrongly', counts: { '3×3': 1, '4×4': 17, '5×5': 41 } },
  { type: 'Other', meaning: 'Rarer tags such as formatting, truncated or looping output', counts: { '3×3': 12, '4×4': 56, '5×5': 30 } },
]

/** 5×5 failures on eigenvalue (296) and determinant (265) problems. */
export const EIGEN_DET_5X5 = 296 + 265
