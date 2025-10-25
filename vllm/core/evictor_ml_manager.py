# SPDX-License-Identifier: Apache-2.0
"""A standalone manager for online learning functionalities."""
import heapq
import json
import os
import queue
import threading
import time
import random
from typing import Dict, List, Tuple

from vllm.core.evictor_ml_model import MLModel
from vllm.core.evictor_profiler import MLProfiler
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

# Debug flag for print statements
ENABLE_DEBUG_PRINTS = True


class OnlineLearningManager:
    """Manages the online learning components, including model training.

    This class is responsible for:
    - Tracking active conversations.
    - Generating training samples (positive and negative).
    - Managing a training queue and a background training thread.
    - Periodically training a machine learning model.
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

        if os.path.exists(self.model_path):
            if ENABLE_DEBUG_PRINTS:
                print(
                    "Loading existing model for online learning from "
                    f"{self.model_path}")
            self.ml_model = MLModel(task="classification", train_mode=False)
            self.ml_model.load_model(self.model_path)
        else:
            if ENABLE_DEBUG_PRINTS:
                print(
                    "Warning: Model path not found. Initializing new model for "
                    "online learning.")
            self.ml_model = MLModel(task="classification", train_mode=True)
        
        self._ml_model_lock = threading.Lock()
        
        # Batch processing for predictions
        self.to_predict_queue = queue.Queue()
        self.batch_size = 16  # Process up to 16 predictions together
        self.batch_timeout = 1  # 1s timeout for batch collection
        
        # Cache for predictions to avoid repeated computation
        # Key: (conv_id, turns), Value: prob_has_next
        self.prediction_cache = {}
        self._prediction_cache_lock = threading.Lock()
        # Max cache size to prevent unbounded growth (can be made configurable)
        self.max_cache_size = self.eviction_algorithm_config.get("max_prediction_cache_size", 10000)
        
        # Profiler
        self.profiler = MLProfiler()
        
        # Start predictor thread (runs only if schedule_prediction is used)
        self.t_predictor_thread = threading.Thread(target=self.predictor_worker,
                                                   daemon=True)
        self.t_predictor_thread.start()

        if self.enable_online_learning:
            self._online_learning_setup()

    def schedule_prediction(self, cache_hint: Dict):
        # only ml eviction algorithm does not have prob_has_next
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
                            print(f"[DEBUG] Prediction cache hit {self.profiler.cache_hits} times")
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
        """Process up to 16 predictions together"""
        with self.profiler.time_operation("process_batch_total"):
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
                
                # Store in prediction cache to avoid recomputation
                conv_id = cache_hint.get('id')
                turns_val = cache_hint.get('turns', 0)
                if conv_id is not None:
                    cache_key = (conv_id, turns_val)
                    with self._prediction_cache_lock:
                        self.prediction_cache[cache_key] = prob
                        
                        # Evict oldest entries if cache is too large
                        # Simple strategy: remove first 10% when limit is exceeded
                        if len(self.prediction_cache) > self.max_cache_size:
                            num_to_remove = max(1, self.max_cache_size // 10)
                            keys_to_remove = list(self.prediction_cache.keys())[:num_to_remove]
                            for key in keys_to_remove:
                                del self.prediction_cache[key]
                            if ENABLE_DEBUG_PRINTS:
                                print(f"[DEBUG] Prediction cache exceeded {self.max_cache_size}, "
                                      f"evicted {num_to_remove} entries. "
                                      f"Current size: {len(self.prediction_cache)}")
                
                self.to_predict_queue.task_done()
            if self.profiler.batch_count % 20 == 19:
                self.profiler.print_stats()

    def _online_learning_setup(self):
        """Initializes components for online model training."""
        self.replay_buffer_size = self.eviction_algorithm_config.get("replay_buffer_size", 16384)
        self.replay_sample_size = self.eviction_algorithm_config.get("replay_sample_size", 128)
        self.prioritize_recent = self.eviction_algorithm_config.get("prioritize_recent", True)
        self.training_interval = self.eviction_algorithm_config.get("training_interval", 2)  # seconds
        self.min_batch_size = self.eviction_algorithm_config.get("min_batch_size", 32)
        self.lr = self.eviction_algorithm_config.get("learning_rate", 1e-3)
        self.conversation_timeout = self.eviction_algorithm_config.get("conversation_timeout", 300)  # 5 minutes
        self.warmup_finished_time = time.time() + self.conversation_timeout

        # Track conversations, not individual sequences
        self.active_conversations: Dict[str, Dict] = {}  # conv_id -> data
        self.conv_heap: List[Tuple[float, str]] = []  # (last_accessed, conv_id)
        self.pending_batch: List[Dict] = []
        self._conv_lock = threading.Lock()

        # --- Replay buffer parameters (can be made configurable) ---
        self.replay_buffer: List[Dict] = []

        self._training_thread = threading.Thread(
            target=self._training_worker_loop, daemon=True)
        self._training_thread.start()

    def _training_worker_loop(self):
        """Worker thread to process training samples and generate batches using a replay buffer.
        To control training frequency, tune self.training_interval (from config)."""
        while True:
            time.sleep(self.training_interval)

            # 1. Generate all available negative samples from timed out conversations and add them to the pending batch.
            self._generate_and_add_timeout_samples()

            # 2. Move all pending samples (positive/negative) to the replay buffer
            with self._conv_lock:
                if len(self.pending_batch) < self.min_batch_size:
                    continue
                
                self.replay_buffer.extend(self.pending_batch)
                self.pending_batch.clear()
                # Keep buffer size constrained
                if len(self.replay_buffer) > self.replay_buffer_size:
                    self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]

            # Do not start training until the warmup period is over.
            if time.time() < self.warmup_finished_time:
                if ENABLE_DEBUG_PRINTS:
                    print(
                        "[DEBUG] Warmup period active. "
                        f"{int(self.warmup_finished_time - time.time())}s "
                        "remaining.")
                continue
            if ENABLE_DEBUG_PRINTS:
                print(f"[DEBUG] Replay buffer size: {len(self.replay_buffer)}")

            # 3. If we have a large enough buffer, sample and train (one step per interval)
            if len(self.replay_buffer) >= self.replay_sample_size:
                if self.prioritize_recent:
                    weights = list(range(len(self.replay_buffer)))
                    batch_to_train = random.choices(self.replay_buffer, weights=weights, k=self.replay_sample_size)
                else:
                    batch_to_train = random.sample(self.replay_buffer, self.replay_sample_size)
                with self._ml_model_lock:
                    loss = self.ml_model.train_online(batch_to_train, lr=self.lr)
                if ENABLE_DEBUG_PRINTS:
                    print(f"[DEBUG] Training Loss: {loss:.4f}")

    def _generate_and_add_timeout_samples(self):
        """Generate negative samples from timed out conversations and add them to the pending batch."""
        now = time.time()
        
        timed_out_samples = []
        with self._conv_lock:
            # Process all conversations that have timed out
            while self.conv_heap and (self.conv_heap[0][0] +
                                      self.conversation_timeout < now):
                timestamp, conv_id = heapq.heappop(self.conv_heap)
                
                # Check for stale heap entries
                if conv_id not in self.active_conversations or \
                   timestamp < self.active_conversations[conv_id]['last_accessed']:
                    continue

                conv_data = self.active_conversations[conv_id]
                timed_out_samples.append({
                    "text": conv_data["text"],
                    "turns": conv_data["turns"],
                    "follow_up": 0
                })
        
        if timed_out_samples:
            with self._conv_lock:
                self.pending_batch.extend(timed_out_samples)

    def on_allocate(self, seq_group: SequenceGroup):
        """to prepare the positive samples and schedule the prediction."""

        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if not waiting_seqs:
            return

        first_seq = waiting_seqs[0]
        if not first_seq.cache_hint:
            return

        # schedule the prediction
        self.schedule_prediction(first_seq.cache_hint)

        if not self.enable_online_learning:
            return

        conv_id = first_seq.cache_hint.get('id')
        if conv_id is None:
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
        """to prepare the negative samples."""
        if not self.enable_online_learning or not seq.cache_hint:
            return

        conv_id = seq.cache_hint.get('id')
        if conv_id is None:
            return

        text = seq.cache_hint.get("conversation_input", "")
        turns = seq.cache_hint.get("turns", 0)

        with self._conv_lock:
            now = time.time()
            self.active_conversations[conv_id] = {
                "text": text,
                "turns": turns,
                "last_accessed": now,
            }
            heapq.heappush(self.conv_heap, (now, conv_id))