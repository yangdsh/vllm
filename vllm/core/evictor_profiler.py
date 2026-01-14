# SPDX-License-Identifier: Apache-2.0
"""Profiler and tracker utilities for cache eviction."""

import time
import statistics
from collections import defaultdict
from typing import Dict, List

import numpy as np


class EvictionProfiler:
    """Simple profiler for identifying hotspots in eviction operations."""
    
    def __init__(self):
        self.times = defaultdict(list)
        self.enabled = True
    
    def time_operation(self, operation_name):
        return ProfilerContext(self, operation_name)
    
    def record_time(self, operation_name, duration):
        if self.enabled:
            self.times[operation_name].append(duration)
    
    def print_stats(self):
        if not self.enabled:
            return
        print("=== Eviction Profiler Stats ===")
        for op, times in self.times.items():
            if times:
                mean_time = statistics.mean(times)
                total_time = sum(times)
                count = len(times)
                print(f"{op}: avg={mean_time*1000:.2f}ms, total={total_time:.3f}s, count={count}")


class ProfilerContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler, operation_name):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.profiler.record_time(self.operation_name, duration)


class MLAccuracyTracker:
    """Tracks ML model prediction accuracy by comparing prob_has_next 
    predictions against ground truth labels from true_tta values.
    """
    
    def __init__(self, threshold: float = 0.5, report_interval: int = 100, 
                 name: str = ""):
        self.predictions = []  # List of probability predictions
        self.true_labels = []  # List of ground truth labels (0 or 1)
        self.threshold = threshold
        self.report_interval = report_interval
        self.total_recorded = 0
        self.name = name  # Optional name to identify tracker in logs
    
    def record(self, prob_has_next: float, true_tta: float):
        """Record a prediction and its ground truth label.
        
        Args:
            prob_has_next: Predicted probability of follow-up (0.0 to 1.0)
            true_tta: True time-to-arrival for next request
        """
        # Calculate ground truth label
        true_label = 1 if true_tta < 1e8 else 0
        
        self.predictions.append(prob_has_next)
        self.true_labels.append(true_label)
        self.total_recorded += 1
        
        # Print stats at regular intervals
        if self.total_recorded % self.report_interval == 0:
            self.print_stats()
    
    def get_metrics(self) -> dict:
        """Calculate accuracy metrics.
        
        Returns:
            Dictionary containing accuracy, precision, recall, F1, and confusion matrix
        """
        if not self.predictions:
            return {}
        
        # Convert probabilities to binary predictions
        pred_labels = [1 if p > self.threshold else 0 for p in self.predictions]
        
        # Calculate confusion matrix
        tp = sum(1 for pred, true in zip(pred_labels, self.true_labels) if pred == 1 and true == 1)
        fp = sum(1 for pred, true in zip(pred_labels, self.true_labels) if pred == 1 and true == 0)
        tn = sum(1 for pred, true in zip(pred_labels, self.true_labels) if pred == 0 and true == 0)
        fn = sum(1 for pred, true in zip(pred_labels, self.true_labels) if pred == 0 and true == 1)
        
        total = tp + fp + tn + fn
        
        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total': total
        }
    
    def print_stats(self):
        """Print formatted accuracy statistics to console."""
        metrics = self.get_metrics()
        
        if not metrics:
            return

        # Suppress noisy block-level logs in production.
        if self.name == "block_level":
            return

        name_suffix = f" [{self.name}]" if self.name else ""
        print(f"\n=== ML Model Accuracy Stats{name_suffix} "
              f"(N={metrics['total']}) ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("Confusion Matrix:")
        print(f"  TP: {metrics['tp']:4d}  FP: {metrics['fp']:4d}")
        print(f"  TN: {metrics['tn']:4d}  FN: {metrics['fn']:4d}")
        print()


class OnlineStatsTracker:
    """Tracks online inference statistics for ML-based eviction."""

    def __init__(self, report_interval: int = 500, name: str = ""):
        self.name = name
        self.report_interval = report_interval

        self.total_predictions: int = 0

        # Per-prediction statistics
        self.probs: List[float] = []
        self.turns: List[int] = []
        self.labels: List[int] = []
        self.true_labels: List[int] = []
        self.correct: List[bool] = []

        # Distributions
        self.label_counts: Dict[int, int] = {0: 0, 1: 0}
        self.turn_distribution: Dict[int, int] = {}

    def record_prediction(self, prob: float, turns: int) -> None:
        """Record a model prediction (without ground-truth label)."""
        self.total_predictions += 1
        self.probs.append(prob)
        self.turns.append(turns)

        self.turn_distribution[turns] = self.turn_distribution.get(turns, 0) + 1

        if self.total_predictions % self.report_interval == 0:
            self.print_summary()

    def record_eval(self, true_label: int, pred_label: int) -> None:
        """Record evaluation info once ground-truth is known."""
        self.labels.append(pred_label)
        self.true_labels.append(true_label)
        self.correct.append(pred_label == true_label)
        self.label_counts[true_label] = self.label_counts.get(true_label, 0) + 1

    def _print_prediction_stats(self) -> None:
        if not self.probs:
            return

        probs = np.asarray(self.probs, dtype=float)
        turns = np.asarray(self.turns, dtype=int) if self.turns else None

        print("\nPrediction Statistics:")
        print(f"  Probabilities: mean={probs.mean():.4f}, std={probs.std():.4f}")
        print(f"  Prob range: [{probs.min():.4f}, {probs.max():.4f}]")

        if turns is not None and turns.size > 0:
            print(f"  Turns: mean={turns.mean():.2f}, "
                  f"range=[{turns.min()}, {turns.max()}]")

    def _print_accuracy_stats(self) -> None:
        if not self.correct:
            return

        correct = np.asarray(self.correct, dtype=bool)
        accuracy = correct.mean() * 100.0

        print("\nAccuracy Statistics:")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        print(f"  Total evaluated: {len(correct)}")

        if self.true_labels:
            true_labels = np.asarray(self.true_labels, dtype=int)
            label_0_mask = true_labels == 0
            label_1_mask = true_labels == 1

            if label_0_mask.any():
                label_0_acc = correct[label_0_mask].mean() * 100.0
            else:
                label_0_acc = 0.0
            if label_1_mask.any():
                label_1_acc = correct[label_1_mask].mean() * 100.0
            else:
                label_1_acc = 0.0

            print(f"  Accuracy for label 0: {label_0_acc:.2f}%")
            print(f"  Accuracy for label 1: {label_1_acc:.2f}%")

    def _print_distribution(self) -> None:
        print("\nData Distribution:")
        total_labels = sum(self.label_counts.values())
        if total_labels > 0:
            label_0_pct = self.label_counts[0] / total_labels * 100.0
            label_1_pct = self.label_counts[1] / total_labels * 100.0
            print(f"  Label 0 (no follow-up): {self.label_counts[0]} "
                  f"({label_0_pct:.2f}%)")
            print(f"  Label 1 (has follow-up): {self.label_counts[1]} "
                  f"({label_1_pct:.2f}%)")

        if self.turn_distribution:
            turn_counts = sorted(self.turn_distribution.items())[:10]
            print(f"  Turn distribution (top 10): {dict(turn_counts)}")

    def print_summary(self) -> None:
        if self.total_predictions == 0:
            return

        header = f"[ONLINE STATS{f' - {self.name}' if self.name else ''}]"
        print("\n" + "=" * 60)
        print(f"{header} After {self.total_predictions} predictions")
        print("=" * 60)

        self._print_prediction_stats()
        self._print_accuracy_stats()
        self._print_distribution()

        print("=" * 60 + "\n")

    def print_final_summary(self) -> None:
        if self.total_predictions == 0:
            return

        print("\n" + "=" * 80)
        print(f"[FINAL ONLINE STATS{f' - {self.name}' if self.name else ''}] "
              "Complete Summary")
        print("=" * 80)

        self.print_summary()

        # Additional final statistics with confusion matrix & F1
        if self.probs and self.true_labels:
            probs = np.asarray(self.probs, dtype=float)
            true_labels = np.asarray(self.true_labels, dtype=int)
            pred_labels = (probs > 0.5).astype(int)

            tn = ((pred_labels == 0) & (true_labels == 0)).sum()
            fp = ((pred_labels == 1) & (true_labels == 0)).sum()
            fn = ((pred_labels == 0) & (true_labels == 1)).sum()
            tp = ((pred_labels == 1) & (true_labels == 1)).sum()

            print("\nConfusion Matrix:")
            print(f"  TN={tn}, FP={fp}")
            print(f"  FN={fn}, TP={tp}")

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            print("\nMetrics:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")

        print("=" * 80 + "\n")


class MLProfiler:
    """Simple profiler for ML predictions."""
    
    def __init__(self):
        self.times = defaultdict(list)
        self.counts = 0
        self.batch_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.enabled = True
    
    def time_operation(self, operation_name):
        return MLProfilerContext(self, operation_name)
    
    def record_time(self, operation_name, duration):
        if self.enabled:
            self.times[operation_name].append(duration)
    
    def print_stats(self):
        if not self.enabled:
            return
        print("=== ML Prediction Profiler Stats ===")
        for op, times in self.times.items():
            if times:
                total_time = sum(times)
                mean_time = total_time / self.counts if self.counts > 0 else 0
                print(f"{op}: avg={mean_time*1000:.2f}ms, total={total_time:.3f}s, count={self.counts}")
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests * 100 if total_requests > 0 else 0
        print(f"Cache: hits={self.cache_hits}, misses={self.cache_misses}, "
              f"hit_rate={hit_rate:.1f}%")


class MLProfilerContext:
    """Context manager for timing ML operations."""
    
    def __init__(self, profiler, operation_name):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.profiler.record_time(self.operation_name, duration)

