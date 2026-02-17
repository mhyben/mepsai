#!/usr/bin/env python3
"""
Improved JSON Evaluation Script
Compares extracted JSON files against ground truth with sophisticated matching.
"""

import argparse
import json
import re
import sys
from os.path import join
from pathlib import Path
from typing import Dict, List, Tuple, Set, Literal

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import InternetAccessLLM


class Evaluator:
    """Evaluates JSON extractions with domain-specific normalization."""

    def __init__(self,
                 ground_truth_path: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 model='qwen/qwen3-vl-8b'):
        """Initialize the evaluator."""
        self.ground_truth_path = Path(ground_truth_path)
        self.prompt_file = "verification_prompt.txt"
        self.llm_client = InternetAccessLLM(prompt_file=self.prompt_file, model=model)

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device='cpu')

        print("Loading ground truth...")
        self.ground_truth = pd.read_csv(self.ground_truth_path)

        # Field mapping
        self.field_mapping = {
            "title": "title",
            "doi": "doi",
            "country": "country",
            "sample_size": "sample_size",
            "sample_type": "sample_type",
            "collection_device": "collection_device",
            "n_timepoints": "n_timepoints",
            "n_days": "n_days",
            "n_nodes": "n_nodes",
            "redundancy": "redundancy",
            "software": "software",
            "estimation_packages": "estimation_packages",
            "network_types_temporal": "network_types",
            "network_types_contemporaneous": "network_types",
            "network_types_between_person": "network_types",
            "centrality_test": "centrality_test",
            "visualisation_packages": "visualisation_packages",
            "preregistered_yn": "preregistered_yn",
            "code_declared_available": "code_declared_available_yn",
            "data_declared_available": "data_declared_available",
            "limitations": "limitations",
            "notes": "notes",
        }

    # ============================================================================
    # COMPARISON FUNCTIONS
    # ============================================================================

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if text1 == text2:
            return 1.0

        if (not text1 or text1 in ["N/A", "N/S"]) and (not text2 or text2 in ["N/A", "N/S"]):
            return 1.0

        if (not text1 or text1 in ["N/A", "N/S"]) or (not text2 or text2 in ["N/A", "N/S"]):
            return 0.0

        # Compute embeddings
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])

        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)

    def _matches_generic(self, gt_val: str, pred_val: str) -> bool:
        """
        Generic string matching with substring containment.
        Returns True if strings are equal or one contains the other.
        """
        norm_gt = _norm_str(gt_val)
        norm_pred = _norm_str(pred_val)

        if norm_gt == norm_pred:
            return True

        # Check substring containment
        if norm_pred and norm_gt and (norm_pred in norm_gt or norm_gt in norm_pred):
            return True

        return False

    def _matches_packages(self, gt_val: str, pred_val: str) -> bool:
        """Match software/package fields by package name overlap."""
        gt_pkgs = _extract_pkg_names(gt_val)
        pred_pkgs = _extract_pkg_names(pred_val)

        if not gt_pkgs or not pred_pkgs:
            return self._matches_generic(gt_val, pred_val)

        # Check if there's any overlap in package names
        if gt_pkgs.intersection(pred_pkgs):
            return True

        return False

    def _matches_semantic(self, gt_val: str, pred_val: str, threshold: float = 0.6) -> bool:
        """Match using semantic similarity for text-heavy fields."""
        if not gt_val or not pred_val:
            return False

        # First try exact/substring match
        if self._matches_generic(gt_val, pred_val):
            return True

        # Then try semantic similarity
        similarity = self._compute_similarity(gt_val, pred_val)
        return similarity >= threshold

    def _verify_mismatch_with_llm(self, field: str, ground_truth: str, predicted: str) -> bool:
        """
        Use LLM to verify if a flagged mismatch is a true mismatch or false positive.
        Returns True if it's a TRUE mismatch (values are different), False if it's a false positive (values match).

        NOTE: This method is deprecated in favor of batch verification.
        """
        if not self.llm_client:
            return True  # If no LLM, assume all mismatches are real

        prompt = f"""Field: {field}
Ground Truth: {ground_truth}
Predicted: {predicted}"""

        try:
            response = self.llm_client.run(
                user_prompt=prompt,
                internet_access=False,
                force_json=True,
                verbose=False
            )

            if response and isinstance(response, dict):
                # The prompt expects {"is_match": true/false, ...}
                if 'is_match' in response:
                    # If LLM says it's a match, it's a FALSE mismatch (return False)
                    # If LLM says it's not a match, it's a TRUE mismatch (return True)
                    return not response['is_match']
                # Fallback to 'same' key if present
                elif 'same' in response:
                    return not response['same']

            # If we can't parse response, assume it's a real mismatch
            return True
        except Exception as e:
            print(f"Warning: LLM verification failed for {field}: {e}")
            return True  # Assume real mismatch on error

    def _verify_mismatches_batch(self, mismatches: List[Dict]) -> Dict[int, bool]:
        """
        Verify multiple mismatches in a single LLM call.
        Returns dict mapping mismatch index to is_true_mismatch boolean.

        Args:
            mismatches: List of mismatch dictionaries

        Returns:
            Dict[int, bool]: Maps index to True (real mismatch) or False (false positive)
        """
        if not self.llm_client or not mismatches:
            return {i: True for i in range(len(mismatches))}

        # Build batch prompt
        pairs_text = ""
        for i, mismatch in enumerate(mismatches):
            pairs_text += (f"Pair {i}: "
                           f"Field: {mismatch['field']} "
                           f"Ground Truth: {mismatch['ground_truth']} "
                           f"Predicted: {mismatch['prediction']}\n")

        try:
            response = self.llm_client.run_rest_api(
                user_prompt=pairs_text,
            )

            if not response or not isinstance(response, dict):
                print("Warning: Invalid batch verification response, treating all as real mismatches")
                return {i: True for i in range(len(mismatches))}

            # Parse results
            results = {}
            for i in range(len(mismatches)):
                idx_str = str(i)
                if idx_str in response and isinstance(response[idx_str], dict):
                    is_match = response[idx_str].get('is_match', False)
                    # If LLM says it's a match, it's a FALSE mismatch (return False)
                    # If LLM says it's not a match, it's a TRUE mismatch (return True)
                    results[i] = not is_match
                else:
                    # If we can't find result for this index, assume real mismatch
                    results[i] = True

            return results

        except Exception as e:
            print(f"Warning: Batch verification failed: {e}")
            return {i: True for i in range(len(mismatches))}

    def _filter_mismatches_with_llm(self, mismatches: List[Dict], batch_size: int = 10) -> Tuple[List[Dict], int]:
        """
        Filter mismatches using LLM verification in batches.
        Returns (filtered_mismatches, num_false_positives).

        Args:
            mismatches: List of mismatch dictionaries
            batch_size: Number of mismatches to verify per LLM call
        """
        if not self.llm_client or not mismatches:
            return mismatches, 0

        print(f"\n🔍 Verifying {len(mismatches)} mismatches with LLM (batch size: {batch_size})...")

        true_mismatches = []
        false_positives = 0

        # Process in batches
        for batch_start in range(0, len(mismatches), batch_size):
            batch_end = min(batch_start + batch_size, len(mismatches))
            batch = mismatches[batch_start:batch_end]

            print(f"  Processing batch {batch_start//batch_size + 1}/{(len(mismatches)-1)//batch_size + 1} ({len(batch)} items)...")

            # Verify batch
            batch_results = self._verify_mismatches_batch(batch)

            # Collect results
            for i, mismatch in enumerate(batch):
                is_true_mismatch = batch_results.get(i, True)

                if is_true_mismatch:
                    true_mismatches.append(mismatch)
                else:
                    false_positives += 1

        print(f"✓ Found {false_positives} false positives out of {len(mismatches)} mismatches")

        return true_mismatches, false_positives

    # ============================================================================
    # EVALUATION
    # ============================================================================

    def _evaluate_file(self, file_id: str, internet_access: bool) -> Dict[str, bool]:
        """
        Evaluate a single file.
        Returns dict of {field_name: is_match}.
        """
        # Load JSON
        json_filename = f"{file_id}.{'internet' if internet_access else 'local'}.json"
        json_path = self.json_dir / json_filename

        if not json_path.exists():
            return {}

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Get ground truth row
        gt_filename = f"{file_id}.txt"
        gt_row = self.ground_truth[self.ground_truth['file'] == gt_filename]

        if gt_row.empty:
            return {}

        gt_row = gt_row.iloc[0]

        results = {}

        for csv_field, json_field in self.field_mapping.items():
            gt_val = gt_row.get(csv_field)
            pred_val = json_data.get(json_field)

            # Skip if both missing
            if (gt_val is None or gt_val == "" or pd.isna(gt_val)) and \
                    (pred_val is None or pred_val == ""):
                continue

            # Network types (boolean comparison)
            if csv_field.startswith("network_types_"):
                if csv_field == "network_types_temporal":
                    nt = "temporal network"
                elif csv_field == "network_types_contemporaneous":
                    nt = "contemporaneous network"
                else:
                    nt = "between-person network"

                gt_bool = _csv_bool(gt_val)
                pred_bool = _network_type_present(pred_val, nt)
                results[csv_field] = (gt_bool == pred_bool)
                continue

            # Boolean fields
            if csv_field in {"redundancy", "centrality_test", "preregistered_yn",
                             "code_declared_available", "data_declared_available"}:
                gt_bool = _csv_bool(gt_val)
                pred_bool = _json_bool(pred_val)
                results[csv_field] = (gt_bool == pred_bool)
                continue

            # DOI
            if csv_field == "doi":
                norm_gt = _norm_doi(gt_val) if gt_val else ""
                norm_pred = _norm_doi(pred_val) if pred_val else ""
                results[csv_field] = (norm_gt == norm_pred)
                continue

            # Numeric fields (n_timepoints, n_days, n_nodes)
            if csv_field in {"n_timepoints", "n_days", "n_nodes"}:
                norm_gt = _norm_numeric_field(gt_val)
                norm_pred = _norm_numeric_field(pred_val)
                results[csv_field] = (norm_gt == norm_pred)
                continue

            # Sample size (may have complex descriptions)
            if csv_field == "sample_size":
                # Try exact numeric match first
                norm_gt = _norm_numeric_field(gt_val)
                norm_pred = _norm_numeric_field(pred_val)
                if norm_gt == norm_pred:
                    results[csv_field] = True
                    continue
                # Fall back to substring matching
                results[csv_field] = self._matches_generic(gt_val, pred_val)
                continue

            # Software/package fields
            if csv_field in {"software", "estimation_packages", "visualisation_packages"}:
                results[csv_field] = self._matches_packages(gt_val, pred_val)
                continue

            # Text-heavy fields (limitations, notes)
            if csv_field in {"limitations", "notes"}:
                results[csv_field] = self._matches_semantic(gt_val, pred_val, threshold=0.5)
                continue

            # Generic string fields
            results[csv_field] = self._matches_generic(gt_val, pred_val)

        return results

    def _evaluate_all(self, internet_used: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Evaluate all files.
        Returns (summary_df, mismatches_list).
        """
        # Extract file IDs
        file_ids = [f.replace('.txt', '') for f in self.ground_truth['file'].values]

        print(f"\nEvaluating {'internet' if internet_used else 'local'} JSON files...")
        print("=" * 100)

        field_stats = {field: {"matches": 0, "total": 0}
                       for field in self.field_mapping.keys()}
        mismatches = []

        total = len(file_ids)
        for idx, file_id in enumerate(file_ids, 1):
            # Progress bar
            _print_progress(idx, total)

            results = self._evaluate_file(file_id, internet_used)

            # Update stats
            for field, is_match in results.items():
                field_stats[field]["total"] += 1
                if is_match:
                    field_stats[field]["matches"] += 1
                else:
                    # Record mismatch
                    gt_filename = f"{file_id}.txt"
                    gt_row = self.ground_truth[self.ground_truth['file'] == gt_filename].iloc[0]
                    json_field = self.field_mapping[field]

                    json_filename = f"{file_id}.{'internet' if internet_used else 'local'}.json"
                    json_path = self.json_dir / json_filename
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                    mismatches.append({
                        "file": gt_filename,
                        "field": field,
                        "ground_truth": str(gt_row.get(field, "")),
                        "prediction": str(json_data.get(json_field, "")),
                    })

        print()  # Newline after progress

        # Build summary dataframe
        summary_rows = []
        for field, stats in field_stats.items():
            total = stats["total"]
            matches = stats["matches"]
            accuracy = (matches / total) if total > 0 else 0.0
            summary_rows.append({
                "Field": field,
                "Matches": matches,
                "Total": total,
                "Accuracy": accuracy,
            })

        summary_df = pd.DataFrame(summary_rows)
        # summary_df = summary_df.sort_values("Accuracy", ascending=False)

        return summary_df, mismatches

    def run(self, file_type: Literal['txt', 'md'] , internet_access: bool = True, verify_with_llm: bool = True, batch_size: int = 10):
        self.json_dir = Path(join('documents', file_type))

        # Evaluate
        summary_df, mismatches = self._evaluate_all(internet_used=internet_access)

        # Verify mismatches with LLM if enabled
        original_mismatch_count = len(mismatches)
        false_positives = 0

        if verify_with_llm and self.llm_client and mismatches:
            mismatches, false_positives = self._filter_mismatches_with_llm(mismatches, batch_size=batch_size)

            # Recalculate accuracy with corrected mismatches
            if false_positives > 0:
                print(f"\n🔄 Recalculating accuracy after removing {false_positives} false positives...")

                # Update summary_df by adding back the false positives to matches
                # We need to distribute them across fields based on which fields had false positives
                # For simplicity, we'll recalculate from scratch
                field_corrections = {}
                for i, mismatch in enumerate(self._evaluate_all(internet_used=internet_access)[1]):
                    if i >= original_mismatch_count:
                        break
                    # Check if this mismatch is not in the filtered list
                    if mismatch not in mismatches:
                        field = mismatch['field']
                        field_corrections[field] = field_corrections.get(field, 0) + 1

                # Update summary_df
                for idx, row in summary_df.iterrows():
                    field = row['Field']
                    if field in field_corrections:
                        corrections = field_corrections[field]
                        summary_df.at[idx, 'Matches'] = row['Matches'] + corrections
                        new_accuracy = summary_df.at[idx, 'Matches'] / summary_df.at[idx, 'Total']
                        summary_df.at[idx, 'Accuracy'] = new_accuracy

        # Print results
        print("\n" + "=" * 100)
        print(f"EVALUATION RESULTS - {'internet' if internet_access else 'local'} FILES")
        if verify_with_llm and self.llm_client and false_positives > 0:
            print(f"(After LLM verification: {false_positives} false positives removed)")
        print("=" * 100)

        _print_markdown_table(summary_df)

        # Overall stats
        total_matches = summary_df['Matches'].sum()
        total_comparisons = summary_df['Total'].sum()
        overall_accuracy = (total_matches / total_comparisons) if total_comparisons > 0 else 0.0

        print(f"\n**Overall Accuracy: {overall_accuracy:.2f}** ({total_matches}/{total_comparisons})")
        print(f"**Total Mismatches: {len(mismatches)}**")
        if verify_with_llm and self.llm_client and false_positives > 0:
            print(f"**False Positives Removed: {false_positives}**")
            print(f"**Original Mismatches: {original_mismatch_count}**")

        # Save mismatches
        if mismatches:
            mismatch_output = f"mismatches_{'internet' if internet_access else 'local'}.csv"
            mismatch_df = pd.DataFrame(mismatches)
            mismatch_df.to_csv(join(self.json_dir, mismatch_output), index=False)
            print(f"\n✓ Mismatch details saved to: {mismatch_output}")

        print("\n" + "=" * 100)
# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def _norm_str(val) -> str:
    """Basic string normalization."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip().lower()
    s = " ".join(s.split())  # Collapse whitespace

    # Remove lightweight punctuation
    for ch in [",", ";"]:
        s = s.replace(ch, "")

    # Normalize simple numbers (14.0 -> 14)
    try:
        if s.replace(".", "", 1).replace("-", "", 1).isdigit():
            num = float(s)
            if num.is_integer():
                s = str(int(num))
    except:
        pass

    return s

def _norm_doi(val: str) -> str:
    """Normalize DOI by removing URL prefixes."""
    v = str(val).strip()
    prefixes = [
        "https://dx.doi.org/",
        "https://doi.org/",
        "http://dx.doi.org/",
        "http://doi.org/",
        "dx.doi.org/",
        "doi.org/",
    ]
    for p in prefixes:
        if v.lower().startswith(p):
            v = v[len(p):]
            break
    return v.lower().strip()

def _extract_pkg_names(val: str) -> Set[str]:
    """Extract package names from software strings."""
    if not val or val in ["N/A", "N/S", ""]:
        return set()

    s = str(val).lower()
    parts = []
    for chunk in s.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)

    names = set()
    for p in parts:
        # Extract leading alphanumeric as package name
        name = []
        for ch in p:
            if ch.isalnum() or ch in {"_", "-"}:
                name.append(ch)
            else:
                break
        if name:
            pkg = "".join(name)
            # Handle common variations
            if pkg in ["lme4", "ime4"]:  # Common OCR error
                names.add("lme4")
            elif pkg == "mlvar":  # Handle mlVAR variants
                names.add("mlvar")
            elif pkg == "mivar":  # Typo variant
                names.add("mlvar")
            else:
                names.add(pkg)

    return names

def _norm_numeric_field(val: str) -> str:
    """
    Normalize numeric fields with optional units.
    Examples:
        '7.0' -> '7'
        '7 times per day' -> '7'
        '10.0' -> '10'
        'once per day' -> '1'
        'daily' -> '1'
    """
    if not val or val in ["N/A", "N/S", ""]:
        return ""

    val_str = str(val).strip().lower()

    # Handle word numbers
    word_nums = {
        'once': '1', 'twice': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
        'nine': '9', 'ten': '10'
    }
    for word, num in word_nums.items():
        if val_str.startswith(word):
            val_str = val_str.replace(word, num, 1)

    # Extract first number
    match = re.search(r'(\d+\.?\d*)', val_str)
    if match:
        num = float(match.group(1))
        if num.is_integer():
            return str(int(num))
        return str(num)

    return val_str

def _csv_bool(val) -> bool:
    """Convert CSV boolean representations."""
    if val is None or pd.isna(val):
        return False
    v = str(val).strip().lower()
    if not v or v in ["n/a", "n/s", "no", "false", "0"]:
        return False
    if v.startswith("yes") or v in {"1", "1.0", "true"}:
        return True
    # "check" is ambiguous - treat as False
    if v == "check":
        return False
    return False

def _json_bool(val) -> bool:
    """Convert JSON boolean representations."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if val is None:
        return False
    v = str(val).strip().lower()
    if v in {"1", "true", "yes"}:
        return True
    if v == "check":  # Treat "check" as ambiguous/false
        return False
    return False

def _network_type_present(predicted_types: str, network_type: str) -> bool:
    """Check if a network type is present in the combined field."""
    if not predicted_types:
        return False
    v = predicted_types.lower()
    return network_type.lower() in v

# ============================================================================
# PRINT FUNCTIONS
# ============================================================================

def _print_progress(current: int, total: int):
    """Print progress bar."""
    if total <= 0:
        return
    bar_len = 30
    frac = current / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(
        f"\rProgress: |{bar}| {current}/{total} ({frac * 100:5.1f}%)"
    )
    sys.stdout.flush()

def _print_markdown_table(df: pd.DataFrame):
    """Print DataFrame as Markdown table."""
    from py_markdown_table.markdown_table import markdown_table

    if df.empty:
        print("No data to display.")
        return

    # Convert DataFrame to list of dictionaries
    table = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                if col == "Accuracy":
                    row_dict[f"{col}:"] = f"{val:.2f}"
                else:
                    row_dict[f"{col}:"] = f"{val:.2f}"
            else:
                row_dict[f"{col}:"] = str(val)
        table.append(row_dict)

    # Generate and print markdown
    markdown = markdown_table(table).set_params(row_sep='markdown').get_markdown()
    print(markdown.replace('```', ''))

if __name__ == '__main__':
    # Initialize evaluator
    evaluator = Evaluator(
        ground_truth_path='ground_truth.csv',
        model_name='all-MiniLM-L6-v2'
    )

    evaluator.run(file_type='txt', internet_access=False)