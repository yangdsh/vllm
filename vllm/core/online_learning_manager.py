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

from vllm.core.learn_conversation import MLModel
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
        self.lr = self.eviction_algorithm_config.get("learning_rate", 5e-4)
        self.training_batch_size = 32
        self.conversation_timeout = 300  # 5 minutes

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

        self.to_predict_queue = queue.Queue()
        self.t_predictor_thread = threading.Thread(target=self.predictor_worker,
                                                   daemon=True)
        self.t_predictor_thread.start()

        if not self.enable_online_learning:
            return

        self._online_learning_setup()

    def schedule_prediction(self, cache_hint: Dict):
        if 'prob_has_next' not in cache_hint:
            self.to_predict_queue.put(cache_hint)

    def predictor_worker(self):
        print("Predictor worker started")
        while True:
            cache_hint = self.to_predict_queue.get()
            if cache_hint is None:
                print("Predictor worker exiting")
                break
            
            with self._ml_model_lock:
                # print('predicting:', cache_hint["conversation_input"])
                prob = self.ml_model.predict_single_processed(
                    cache_hint["conversation_input"], cache_hint["turns"])
            cache_hint["prob_has_next"] = prob
            self.to_predict_queue.task_done()

    def _online_learning_setup(self):
        """Initializes components for online model training."""
        self.warmup_finished_time = time.time() + self.conversation_timeout

        # Track conversations, not individual sequences
        self.active_conversations: Dict[str, Dict] = {}  # conv_id -> data
        self.conv_heap: List[Tuple[float, str]] = []  # (last_accessed, conv_id)
        self.pending_batch: List[Dict] = []
        self._conv_lock = threading.Lock()

        # --- Replay buffer parameters (can be made configurable) ---
        self.replay_buffer: List[Dict] = []
        self.replay_buffer_size = self.eviction_algorithm_config.get("replay_buffer_size", 16384)
        self.replay_sample_size = self.eviction_algorithm_config.get("replay_sample_size", 128)
        self.prioritize_recent = self.eviction_algorithm_config.get("prioritize_recent", True)
        self.training_interval = self.eviction_algorithm_config.get("training_interval", 10)  # seconds

        self._training_thread = threading.Thread(
            target=self._training_worker_loop, daemon=True)
        self._training_thread.start()

    def _training_worker_loop(self):
        """Worker thread to process training samples and generate batches using a replay buffer.
        To control training frequency, tune self.training_interval (from config)."""
        while True:
            time.sleep(self.training_interval)

            # 1. Generate all available negative samples from timed out conversations and add them to the pending batch.
            self._generate_all_available_timeout_samples(self.pending_batch)

            # 2. Move all pending samples (positive/negative) to the replay buffer
            with self._conv_lock:
                if self.pending_batch:
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
                    self.ml_model.train_online(batch_to_train, lr=self.lr)
                if ENABLE_DEBUG_PRINTS:
                    print("[DEBUG] Training complete on replay buffer batch.")

    def _generate_all_available_timeout_samples(self, batch: List[Dict]):
        """Generate negative samples from timed out conversations."""
        now = time.time()
        with self._conv_lock:
            # Process all conversations that have timed out
            while self.conv_heap and (self.conv_heap[0][0] +
                                      self.conversation_timeout < now):
                timestamp, conv_id = heapq.heappop(self.conv_heap)
                if conv_id in self.active_conversations:
                    # If the timestamp in the heap is stale (i.e., the
                    # conversation has been updated since this heap entry was
                    # created), ignore this entry.
                    if timestamp < self.active_conversations[conv_id]['last_accessed']:
                        continue

                    conv_data = self.active_conversations[conv_id]
                    batch.append({
                        "text": conv_data["text"],
                        "turns": conv_data["turns"],
                        "follow_up": 0
                    })

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

        if not self.enable_online_learning or time.time() < self.warmup_finished_time:
            return

        conv_id = first_seq.cache_hint['id']
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
        if not self.enable_online_learning:
            return

        conv_id = seq.cache_hint['id']
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