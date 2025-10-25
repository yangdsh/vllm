# SPDX-License-Identifier: Apache-2.0
"""Profiler and tracker utilities for cache eviction."""

import time
import statistics
from collections import defaultdict


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
    
    def __init__(self, threshold: float = 0.5, report_interval: int = 100):
        self.predictions = []  # List of probability predictions
        self.true_labels = []  # List of ground truth labels (0 or 1)
        self.threshold = threshold
        self.report_interval = report_interval
        self.total_recorded = 0
    
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
        
        print(f"\n=== ML Model Accuracy Stats (N={metrics['total']}) ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Confusion Matrix:")
        print(f"  TP: {metrics['tp']:4d}  FP: {metrics['fp']:4d}")
        print(f"  TN: {metrics['tn']:4d}  FN: {metrics['fn']:4d}")
        print()


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

