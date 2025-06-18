# SPDX-License-Identifier: Apache-2.0
"""A standalone manager for online learning functionalities."""
import heapq
import json
import os
import queue
import threading
import time
from typing import Dict, List, Tuple

from vllm.core.learn_conversation import MLModel
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

# Debug flag for print statements
ENABLE_DEBUG_PRINTS = False


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

        if not self.enable_online_learning:
            return

        self._online_learning_setup()

    def _online_learning_setup(self):
        """Initializes components for online model training."""
        self.training_queue: queue.Queue[Dict] = queue.Queue()
        self.training_batch_size = 32
        self.conversation_timeout = 30  # 0.5 minutes
        self.warmup_finished_time = time.time() + self.conversation_timeout

        # Track conversations, not individual sequences
        self.active_conversations: Dict[str, Dict] = {}  # conv_id -> data
        self.conv_heap: List[Tuple[float,
                                   str]] = []  # (last_accessed, conv_id)
        self.pending_batch: List[Dict] = []
        self._conv_lock = threading.Lock()

        model_path = "checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_epoch20_metric_neg0_0036.pt"
        if os.path.exists(model_path):
            if ENABLE_DEBUG_PRINTS:
                print(
                    "Loading existing model for online learning from "
                    f"{model_path}")
            self.ml_model = MLModel(task="classification", train_mode=False)
            self.ml_model.load_model(model_path)
        else:
            if ENABLE_DEBUG_PRINTS:
                print(
                    "Warning: Model path not found. Initializing new model for "
                    "online learning.")
            self.ml_model = MLModel(task="classification", train_mode=True)

        self._stop_event = threading.Event()
        self._training_thread = threading.Thread(
            target=self._training_worker_loop, daemon=True)
        self._training_thread.start()
        if ENABLE_DEBUG_PRINTS:
            print("Online learning worker thread started.")
            print(
                "Training is frozen for a "
                f"{self.conversation_timeout}s warmup period.")

    def stop_worker(self):
        """Stops the background training thread."""
        if not self.enable_online_learning:
            return
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, '_training_thread') and self._training_thread.is_alive(
        ):
            self._training_thread.join()

    def _training_worker_loop(self):
        """Worker thread to process training samples and generate batches."""
        TRAINING_INTERVAL = 10  # seconds

        while not self._stop_event.is_set():
            if ENABLE_DEBUG_PRINTS:
                print(
                    "[DEBUG] Training worker loop iteration. Sleeping for "
                    f"{TRAINING_INTERVAL}s.")
            time.sleep(TRAINING_INTERVAL)

            # Do not start training until the warmup period is over.
            if time.time() < self.warmup_finished_time:
                if ENABLE_DEBUG_PRINTS:
                    print(
                        "[DEBUG] Warmup period active. "
                        f"{int(self.warmup_finished_time - time.time())}s "
                        "remaining.")
                continue

            batch = self.pending_batch

            # 1. Drain all positive samples from the queue
            num_pos_samples = 0
            while not self.training_queue.empty():
                try:
                    positive_sample = self.training_queue.get_nowait()
                    batch.append(positive_sample)
                    num_pos_samples += 1
                    self.training_queue.task_done()
                except queue.Empty:
                    break
            if ENABLE_DEBUG_PRINTS:
                print(
                    "[DEBUG] Drained {num_pos_samples} positive samples from "
                    "the queue.")

            # 2. Generate all available negative samples
            self._generate_all_available_timeout_samples(batch)

            # 3. If we have a large enough batch, train on it. Otherwise, hold.
            if ENABLE_DEBUG_PRINTS:
                print(f"[DEBUG] Current batch size: {len(batch)}")
            if len(batch) >= self.training_batch_size:
                if ENABLE_DEBUG_PRINTS:
                    print(f"[DEBUG] Training on batch of size {len(batch)}")
                self.ml_model.train_step(batch)
                batch.clear()
                if ENABLE_DEBUG_PRINTS:
                    print("[DEBUG] Training complete. Batch cleared.")

    def _generate_all_available_timeout_samples(self, batch: List[Dict]):
        """Generate negative samples from timed out conversations."""
        now = time.time()
        with self._conv_lock:
            # Process all conversations that have timed out
            while self.conv_heap and (self.conv_heap[0][0] +
                                      self.conversation_timeout < now):
                _, conv_id = heapq.heappop(self.conv_heap)
                if conv_id in self.active_conversations:
                    conv_data = self.active_conversations.pop(conv_id)
                    if ENABLE_DEBUG_PRINTS:
                        print(
                            "[DEBUG] Generating negative sample from timed out"
                            f" conversation {conv_id}")
                    batch.append({
                        'conv_id': conv_id,
                        'label': 0,  # Negative sample
                        'data': conv_data
                    })

    def on_allocate(self, seq_group: SequenceGroup):
        """Callback to be called when a sequence group is allocated."""
        if not self.enable_online_learning:
            return

        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if not waiting_seqs:
            return

        first_seq = waiting_seqs[0]
        if first_seq.cache_hint and 'id' in first_seq.cache_hint:
            conv_id = first_seq.cache_hint['id']
            text = first_seq.cache_hint.get("conversation_input", "")
            turns = first_seq.cache_hint.get("turns", 0)

            with self._conv_lock:
                now = time.time()
                self.active_conversations[conv_id] = {
                    "text": text,
                    "turns": turns,
                    "last_accessed": now,
                }
                heapq.heappush(self.conv_heap, (now, conv_id))

            if ENABLE_DEBUG_PRINTS:
                print(
                    "[DEBUG] Tracking conversation for online learning: "
                    f"{conv_id}")

            # Generate a positive sample if this is a follow-up turn
            if turns > 0:
                positive_sample = {
                    "text": text,
                    "turns": turns,
                    "follow_up": 1
                }
                if ENABLE_DEBUG_PRINTS:
                    print(
                        "Generated positive sample for conv_id: "
                        f"{conv_id} (turns={turns})")
                self.training_queue.put(positive_sample)

    def on_free(self, seq: Sequence):
        """Callback to be called when a sequence is freed."""
        if not self.enable_online_learning:
            return

        with self._conv_lock:
            if seq.cache_hint and 'id' in seq.cache_hint:
                conv_id = seq.cache_hint['id']
                if conv_id in self.active_conversations:
                    # Update last_accessed time as this request is ending.
                    now = time.time()
                    self.active_conversations[conv_id]['last_accessed'] = now
                    heapq.heappush(self.conv_heap, (now, conv_id)) 