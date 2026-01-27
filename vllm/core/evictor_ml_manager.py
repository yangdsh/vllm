# SPDX-License-Identifier: Apache-2.0
"""A standalone manager for online learning functionalities.

This module provides hidden state extraction for use as ML model embeddings.
Hidden states from the last transformer layer are used as input features
for the ML-based cache eviction model.
"""
import heapq
import json
import os
import queue
import threading
import time
import random
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

from vllm.core.evictor_ml_model import MLModel
from vllm.core.evictor_profiler import (MLProfiler, MLAccuracyTracker,
                                        OnlineStatsTracker)
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

logger = init_logger(__name__)

# Debug flag for print statements
ENABLE_DEBUG_PRINTS = True


class TrainingDataCollector:
    """Collects hidden states and labels for offline training.
    
    Uses true labels from cache_hint metadata (true_tta field) to determine
    whether a conversation turn will have a follow-up or not.
    """
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # File paths
        self.hidden_states_file = os.path.join(output_dir, "hidden_states.npy")
        self.labels_file = os.path.join(output_dir, "labels.jsonl")
        
        self._lock = threading.Lock()
        
        # Counters
        self.num_samples = 0
        self.num_positive = 0
        self.num_negative = 0
        
        # Batch for writing
        self._hidden_states_batch: List[np.ndarray] = []
        self._labels_batch: List[Dict] = []
        self._batch_size = 100  # Write to disk every N samples
        
        print(f"[DataCollector] Initialized. Output dir: {output_dir}")
    
    def log_sample_direct(self, conv_id: str, turns: int, label: int,
                           hidden_state: torch.Tensor, text: str = ""):
        """Log a training sample directly with hidden state and label.
        
        This is the preferred method for data collection as it uses the true
        label from cache_hint metadata instead of waiting for follow-up.
        
        Args:
            conv_id: Conversation ID
            turns: Turn number in conversation
            label: 1 = has follow-up, 0 = no follow-up
            hidden_state: Hidden state tensor from model
            text: Optional conversation text for reference
        """
        if hidden_state is None:
            logger.debug(f"No hidden state for conv_id={conv_id}, turns={turns}")
            return
            
        with self._lock:
            # Convert to numpy
            if isinstance(hidden_state, torch.Tensor):
                hs_numpy = hidden_state.cpu().numpy()
            else:
                hs_numpy = hidden_state
            
            # Prepare the sample
            sample_metadata = {
                "conv_id": conv_id,
                "turns": turns,
                "label": label,  # 1 = has follow-up, 0 = no follow-up
                "text": text[:200] if text else "",  # Truncate for reference
                "hidden_state_shape": list(hs_numpy.shape),
            }
            
            self._hidden_states_batch.append(hs_numpy)
            self._labels_batch.append(sample_metadata)
            
            self.num_samples += 1
            if label == 1:
                self.num_positive += 1
            else:
                self.num_negative += 1
            
            # Write to disk periodically
            if len(self._labels_batch) >= self._batch_size:
                self._flush_to_disk()
            
            if self.num_samples % 100 == 0:
                print(f"[DataCollector] Collected {self.num_samples} samples "
                      f"(+:{self.num_positive}, -:{self.num_negative})")
    
    def _flush_to_disk(self):
        """Write batched data to disk."""
        if not self._labels_batch:
            return
        
        # Append hidden states to numpy file
        hidden_states_array = np.stack(self._hidden_states_batch)
        
        if os.path.exists(self.hidden_states_file):
            # Load existing and append
            existing = np.load(self.hidden_states_file)
            combined = np.concatenate([existing, hidden_states_array], axis=0)
            np.save(self.hidden_states_file, combined)
        else:
            np.save(self.hidden_states_file, hidden_states_array)
        
        # Append labels to JSONL file
        with open(self.labels_file, 'a') as f:
            for sample in self._labels_batch:
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"Flushed {len(self._labels_batch)} samples to disk")
        
        # Clear batches
        self._hidden_states_batch.clear()
        self._labels_batch.clear()
    
    def finalize(self):
        """Flush remaining data and print summary."""
        with self._lock:
            self._flush_to_disk()
        
        print(f"\n[DataCollector] Final Summary:")
        print(f"  Total samples: {self.num_samples}")
        print(f"  Positive (has follow-up): {self.num_positive}")
        print(f"  Negative (no follow-up): {self.num_negative}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Hidden states: {self.hidden_states_file}")
        print(f"  Labels: {self.labels_file}")


class OnlineLearningManager:
    """Manages the online learning components, including model training.

    This class is responsible for:
    - Tracking active conversations.
    - Generating training samples (positive and negative).
    - Managing a training queue and a background training thread.
    - Periodically training a machine learning model.
    - Extracting hidden states from model output for ML predictions.
    """

    def __init__(self, eviction_algorithm_config: str = ''):
        if eviction_algorithm_config:
            self.eviction_algorithm_config = json.loads(
                eviction_algorithm_config)
        else:
            self.eviction_algorithm_config = {}

        self.enable_online_learning = self.eviction_algorithm_config.get(
            "enable_online_learning", False)
        self.model_path = self.eviction_algorithm_config.get("model_path", "")
        
        # Check if we should use hidden state embeddings
        self.use_hidden_state = self.eviction_algorithm_config.get(
            "use_hidden_state", False)
        
        # Data collection for offline training
        self.enable_data_collection = self.eviction_algorithm_config.get(
            "enable_data_collection", False)
        self.data_collection_dir = self.eviction_algorithm_config.get(
            "data_collection_dir", "training_data")
        
        if self.enable_data_collection:
            self.data_collector = TrainingDataCollector(
                output_dir=self.data_collection_dir)
            print(f"[OnlineLearningManager] Data collection enabled. "
                  f"Output: {self.data_collection_dir}")
        else:
            self.data_collector = None

        if os.path.exists(self.model_path):
            print(
                "Loading existing model for online learning from "
                f"{self.model_path}")
            if self.use_hidden_state:
                self.ml_model = MLModel(
                    task="classification", 
                    train_mode=False,
                    use_hidden_state_embeddings=True
                )
            else:
                self.ml_model = MLModel(task="classification", train_mode=False)
            self.ml_model.load_model(self.model_path)
        else:
            print(
                "Warning: Model path not found. Initializing new model for "
                "online learning.")
            if self.use_hidden_state:
                self.ml_model = MLModel(
                    task="classification", 
                    train_mode=True,
                    use_hidden_state_embeddings=True,
                    embedding_dim=4096 # todo: should be set by the LLM model
                )
            else:
                self.ml_model = MLModel(task="classification", train_mode=True)
        
        self._ml_model_lock = threading.Lock()
        
        # Batch processing for predictions
        self.to_predict_queue = queue.Queue()
        self.batch_size = 16  # Process up to 16 predictions together
        self.batch_timeout = 0.001  # 0.001s timeout for batch collection
        
        # Cache for predictions to avoid repeated computation
        # Key: (conv_id, turns), Value: prob_has_next
        self.prediction_cache = {}
        self._prediction_cache_lock = threading.Lock()
        # Max cache size to prevent unbounded growth (can be made configurable)
        self.max_cache_size = self.eviction_algorithm_config.get(
            "max_prediction_cache_size", 10000)
        
        # Profiler
        self.profiler = MLProfiler()
        
        # Accuracy tracker at sequence level (prediction should be ready by on_free)
        # Compare with block-level tracker in evictor to see prediction completion rate
        self.accuracy_tracker = MLAccuracyTracker(
            threshold=0.5, report_interval=100, name="seq_level")
        
        # Online inference statistics tracking
        self.online_stats = OnlineStatsTracker(
            report_interval=500, name="seq_level_online")
        
        # Start predictor thread (runs only if schedule_prediction is used)
        self.t_predictor_thread = threading.Thread(target=self.predictor_worker,
                                                   daemon=True)
        self.t_predictor_thread.start()

        if self.enable_online_learning:
            self._online_learning_setup()

    def schedule_prediction(self, cache_hint: Dict):
        """Schedule a prediction for a cache hint if not already computed."""
        # Only ml eviction algorithm does not have prob_has_next
        if 'prob_has_next' in cache_hint:
            return
        
        with self.profiler.time_operation("schedule_prediction"):
            # Check if we already have a prediction for this (conv_id, turns) pair
            conv_id = cache_hint.get('id')
            turns = cache_hint.get('turns', 0)
            
            if conv_id is not None:
                cache_key = (conv_id, turns)
                with self._prediction_cache_lock:
                    if cache_key in self.prediction_cache:
                        # Use cached prediction
                        cache_hint['prob_has_next'] = self.prediction_cache[cache_key]
                        self.profiler.cache_hits += 1
                        if ENABLE_DEBUG_PRINTS and self.profiler.cache_hits % 100 == 0:
                            print(f"[DEBUG] Prediction cache hit "
                                  f"{self.profiler.cache_hits} times")
                        return
            
            # Schedule for prediction if not in cache
            self.profiler.cache_misses += 1
            self.to_predict_queue.put(cache_hint)

    def clear_conversation_cache(self, conv_id: str):
        """Clear all cached predictions for a specific conversation.
        
        This can be called when a conversation is completed to free up cache space.
        """
        if conv_id is None:
            return
        
        with self._prediction_cache_lock:
            # Find and remove all entries with this conv_id
            keys_to_remove = [key for key in self.prediction_cache.keys() 
                             if key[0] == conv_id]
            for key in keys_to_remove:
                del self.prediction_cache[key]
            
            if ENABLE_DEBUG_PRINTS and keys_to_remove:
                print(f"[DEBUG] Cleared {len(keys_to_remove)} cache entries "
                      f"for conversation {conv_id}")

    def predictor_worker(self):
        """Background worker thread for batch prediction."""
        print("Predictor worker started")
        
        while True:
            batch = []
            start_time = time.time()
            
            # Collect batch of predictions
            while len(batch) < self.batch_size:
                timeout = self.batch_timeout - (time.time() - start_time)
                if timeout <= 0:
                    break
                    
                try:
                    cache_hint = self.to_predict_queue.get(timeout=timeout)
                    if cache_hint is None:
                        print("Predictor worker exiting")
                        return
                    batch.append(cache_hint)
                except queue.Empty:
                    break
            
            # Process batch if we have items
            if batch:
                self._process_prediction_batch(batch)

    def _process_prediction_batch(self, batch: List[Dict]):
        """Process a batch of predictions using hidden states or text."""
        with self.profiler.time_operation("process_batch_total"):
            # Prefer hidden states if available for all items in the batch.
            use_hidden_states = any(
                ch.get("hidden_state") is not None for ch in batch
            )

            if use_hidden_states:
                # Extract hidden states from cache_hints
                hidden_states_list = []
                turns = []
                for ch in batch:
                    hidden_state = ch.get("hidden_state")
                    if hidden_state is not None:
                        hidden_states_list.append(hidden_state)
                    else:
                        hidden_states_list.append(None)
                        logger.warning(f"No hidden state for conv_id={ch.get('id')}, "
                                       f"turns={ch.get('turns', 0)}")
                    turns.append(ch.get("turns", 0))
                
                # If all have hidden states, use them for prediction
                if all(hs is not None for hs in hidden_states_list):
                    probs = None
                    with self._ml_model_lock:
                        try:
                            # Use batch prediction with hidden states.
                            probs = self.ml_model.predict_batch_processed(
                                turns=turns,
                                hidden_states_list=hidden_states_list,
                            )
                            self.profiler.counts += len(batch)
                            self.profiler.batch_count += 1
                        except Exception as e:
                            print(f"ML prediction error with hidden states: {e}")
                            use_hidden_states = False
                else:
                    use_hidden_states = False

            if not use_hidden_states:
                # Use text-based prediction (original behavior)
                texts = []
                turns = []
                for ch in batch:
                    texts.append(ch.get("conversation_input", ""))
                    turns.append(ch.get("turns", 0))

                probs = None
                with self._ml_model_lock:
                    try:
                        probs = self.ml_model.predict_batch_processed(texts, turns)
                        self.profiler.counts += len(batch)
                        self.profiler.batch_count += 1
                    except Exception as e:
                        print(f"ML batch prediction error: {e}")
                        # If prediction fails, use default probabilities
                        probs = [0.2] * len(batch)

            # Attach results back to cache_hints and store in cache
            for cache_hint, prob in zip(batch, probs):
                cache_hint["prob_has_next"] = prob
                
                # Track prediction statistics for online inference
                turns_val = cache_hint.get('turns', 0)
                self.online_stats.record_prediction(prob, turns_val)
                
                # Store in prediction cache to avoid recomputation
                conv_id = cache_hint.get('id')
                turns_val = cache_hint.get('turns', 0)
                if conv_id is not None:
                    cache_key = (conv_id, turns_val)
                    with self._prediction_cache_lock:
                        self.prediction_cache[cache_key] = prob
                        
                        # Evict oldest entries if cache is too large
                        if len(self.prediction_cache) > self.max_cache_size:
                            num_to_remove = max(1, self.max_cache_size // 10)
                            keys_to_remove = list(
                                self.prediction_cache.keys())[:num_to_remove]
                            for key in keys_to_remove:
                                del self.prediction_cache[key]
                            if ENABLE_DEBUG_PRINTS:
                                print(f"[DEBUG] Prediction cache exceeded "
                                      f"{self.max_cache_size}, evicted "
                                      f"{num_to_remove} entries.")
                
                self.to_predict_queue.task_done()
            
            if self.profiler.batch_count % 100 == 99:
                self.profiler.print_stats()

    def _online_learning_setup(self):
        """Initializes components for online model training."""
        self.replay_buffer_size = self.eviction_algorithm_config.get(
            "replay_buffer_size", 16384)
        self.replay_sample_size = self.eviction_algorithm_config.get(
            "replay_sample_size", 128)
        self.prioritize_recent = self.eviction_algorithm_config.get(
            "prioritize_recent", True)
        self.training_interval = self.eviction_algorithm_config.get(
            "training_interval", 2)  # seconds
        self.min_batch_size = self.eviction_algorithm_config.get(
            "min_batch_size", 32)
        self.lr = self.eviction_algorithm_config.get("learning_rate", 1e-3)
        self.conversation_timeout = self.eviction_algorithm_config.get(
            "conversation_timeout", 300)  # 5 minutes
        self.warmup_finished_time = time.time() + self.conversation_timeout

        # Track conversations, not individual sequences
        self.active_conversations: Dict[str, Dict] = {}  # conv_id -> data
        self.conv_heap: List[Tuple[float, str]] = []  # (last_accessed, conv_id)
        self.pending_batch: List[Dict] = []
        self._conv_lock = threading.Lock()

        # Replay buffer
        self.replay_buffer: List[Dict] = []

        self._training_thread = threading.Thread(
            target=self._training_worker_loop, daemon=True)
        self._training_thread.start()

    def _training_worker_loop(self):
        """Worker thread for online training using a replay buffer."""
        while True:
            time.sleep(self.training_interval)

            # 1. Generate negative samples from timed out conversations
            self._generate_and_add_timeout_samples()

            # 2. Move pending samples to replay buffer
            with self._conv_lock:
                if len(self.pending_batch) < self.min_batch_size:
                    continue
                
                self.replay_buffer.extend(self.pending_batch)
                self.pending_batch.clear()
                # Keep buffer size constrained
                if len(self.replay_buffer) > self.replay_buffer_size:
                    self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]

            # Wait for warmup period
            if time.time() < self.warmup_finished_time:
                if ENABLE_DEBUG_PRINTS:
                    print(
                        "[DEBUG] Warmup period active. "
                        f"{int(self.warmup_finished_time - time.time())}s "
                        "remaining.")
                continue
            
            if ENABLE_DEBUG_PRINTS:
                print(f"[DEBUG] Replay buffer size: {len(self.replay_buffer)}")

            # 3. Sample and train
            if len(self.replay_buffer) >= self.replay_sample_size:
                if self.prioritize_recent:
                    weights = list(range(len(self.replay_buffer)))
                    batch_to_train = random.choices(
                        self.replay_buffer, weights=weights, 
                        k=self.replay_sample_size)
                else:
                    batch_to_train = random.sample(
                        self.replay_buffer, self.replay_sample_size)
                with self._ml_model_lock:
                    loss = self.ml_model.train_online(batch_to_train, lr=self.lr)
                if ENABLE_DEBUG_PRINTS:
                    print(f"[DEBUG] Training Loss: {loss:.4f}")

    def _generate_and_add_timeout_samples(self):
        """Generate negative samples from timed out conversations.
        
        Note: This is only used for online learning. Data collection for
        offline training uses true labels directly in on_free().
        """
        now = time.time()
        
        timed_out_samples = []
        
        with self._conv_lock:
            while self.conv_heap and (self.conv_heap[0][0] +
                                      self.conversation_timeout < now):
                timestamp, conv_id = heapq.heappop(self.conv_heap)
                
                # Check for stale heap entries
                if conv_id not in self.active_conversations or \
                   timestamp < self.active_conversations[conv_id]['last_accessed']:
                    continue

                conv_data = self.active_conversations.pop(conv_id)
                timed_out_samples.append({
                    "text": conv_data["text"],
                    "turns": conv_data["turns"],
                    "follow_up": 0
                })
        
        if timed_out_samples:
            with self._conv_lock:
                self.pending_batch.extend(timed_out_samples)

    def on_allocate(self, seq_group: SequenceGroup):
        """Called when a sequence group is allocated.
        
        Schedules prediction and prepares samples for online learning.
        Note: Data collection is handled in on_free() using true labels.
        
        For hidden state models, prediction is scheduled AFTER prefill completes
        (in extract_hidden_state_for_seq) since we need the hidden state first.
        For text embedding models, prediction is scheduled here immediately.
        """
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if not waiting_seqs:
            return

        first_seq = waiting_seqs[0]
        if not first_seq.cache_hint:
            return

        # Only schedule prediction here for text embedding models
        # Hidden state models schedule prediction after extraction in extract_hidden_state_for_seq
        if not getattr(self.ml_model, 'use_hidden_state_embeddings', False):
            self.schedule_prediction(first_seq.cache_hint)

        conv_id = first_seq.cache_hint.get('id')
        if conv_id is None:
            return
        
        current_turns = first_seq.cache_hint.get('turns', 0)

        if not self.enable_online_learning:
            return
            
        # Generate a positive sample if this is a follow-up turn
        with self._conv_lock:
            if conv_id in self.active_conversations:
                conv_data = self.active_conversations.pop(conv_id)
                positive_sample = {
                    "text": conv_data["text"],
                    "turns": conv_data["turns"],
                    "follow_up": 1
                }
                self.pending_batch.append(positive_sample)

    def on_free(self, seq: Sequence):
        """Called when a sequence is freed.
        
        Logs training samples for data collection using the true label from
        cache_hint metadata, prepares samples for online learning, and
        records ML accuracy at the sequence level (after prediction is ready).
        """
        if not seq.cache_hint:
            return

        conv_id = seq.cache_hint.get('id')
        if conv_id is None:
            return

        text = seq.cache_hint.get("conversation_input", "")
        turns = seq.cache_hint.get("turns", 0)
        hidden_state = seq.cache_hint.get("hidden_state")
        
        # Track ML accuracy at sequence level (prediction should be ready by now)
        # This is more reliable than block-level tracking which races with prediction
        if 'prob_has_next' in seq.cache_hint and 'true_tta' in seq.cache_hint:
            prob = seq.cache_hint['prob_has_next']
            true_tta = seq.cache_hint['true_tta']
            self.accuracy_tracker.record(prob, true_tta)
            
            # Track prediction accuracy and labels
            true_label = 1 if true_tta < 1e8 else 0
            pred_label = 1 if prob > 0.5 else 0
            self.online_stats.record_eval(true_label, pred_label)
        
        # Data collection: log sample directly with true label from metadata
        # true_tta < 1e8 means there will be a follow-up (label=1)
        # true_tta >= 1e8 means no follow-up (label=0)
        if self.data_collector and hidden_state is not None:
            true_tta = seq.cache_hint.get("true_tta", float('inf'))
            true_label = 1 if true_tta < 1e8 else 0
            self.data_collector.log_sample_direct(
                conv_id=conv_id,
                turns=turns,
                label=true_label,
                hidden_state=hidden_state,
                text=text
            )

        if not self.enable_online_learning:
            return

        with self._conv_lock:
            now = time.time()
            self.active_conversations[conv_id] = {
                "text": text,
                "turns": turns,
                "last_accessed": now,
            }
            heapq.heappush(self.conv_heap, (now, conv_id))
    
    def __del__(self):
        """Print final statistics when manager is destroyed."""
        try:
            if hasattr(self, 'online_stats') and \
               self.online_stats.total_predictions > 0:
                self.online_stats.print_final_summary()
        except:
            pass  # Ignore errors during cleanup
