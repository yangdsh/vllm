# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
import json
import time
import statistics
import numpy as np
import threading
from collections import defaultdict, deque
from sortedcontainers import SortedDict
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vllm.core.evictor_profiler import EvictionProfiler, MLAccuracyTracker

def probability_of_future_arrival(prob_has_next, exp_scale, elapsed_time, debug=False):
    if prob_has_next == 0 or exp_scale == 0:
        return 0.0
    # print(f"prob_has_next: {prob_has_next}, exp_scale: {exp_scale}, elapsed_time: {elapsed_time}")
    prob_not_accessed_till_now = np.exp(-elapsed_time / exp_scale)
    return (prob_has_next * prob_not_accessed_till_now) / (
            prob_has_next * prob_not_accessed_till_now + (1 - prob_has_next)
    )

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()
    S3FIFO = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass

class CacheStat:
    def __init__(self):
        self.stat = defaultdict(list)
        self.average = {}
        self.last_log_time = 0
        self.log_interval = 10000  # Log every 5 seconds
    
    def get_average(self, key):
        if key in self.average:
            return self.average[key]
        else:
            return 1
    
    def append(self, key, value):
        if key not in self.average:
            self.average[key] = 0
        self.average[key] = (self.average[key] * len(self.stat[key]) + value) / (len(self.stat[key]) + 1)
        self.stat[key].append(value)
    
    def summary(self):
        current_time = time.time()
        if current_time - self.last_log_time < self.log_interval:
            return 0
        self.last_log_time = current_time
        has_data = 0

        for key in self.stat.keys():
            data = self.stat[key]
            if not data:
                continue
            
            mean = statistics.mean(data)
            std_dev = statistics.stdev(data) if len(data) > 1 else 0
            print(f"Summary for {key}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")
            has_data = 1
        return has_data


class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, cache_hint: dict = None, score: float = 0):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.cache_hint = cache_hint
        self.score = score

class LRUMLEvictor(Evictor):

    def __init__(self, config):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.sorted_dict = SortedDict()
        self.id_to_last_access = {}
        self.id_to_first_access = {}
        self.to_delete_blocks = []
        self.config = self.parse_str_to_dict(config)
        self.stat = CacheStat()
        
        # Increased refresh interval
        self.last_refresh_time = time.time()
        self.refresh_count = 0
        self.INSPECT_INTERVAL = 10  # Increased from 5 to reduce overhead
        
        # Background threading for rescoring
        self._rescore_lock = threading.Lock()
        self._rescore_thread = None
        
        # Track largest seen last_accessed for simulation time
        self._max_last_accessed = 0.0
        
        # Profiler
        self.profiler = EvictionProfiler()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def parse_str_to_dict(self, s: str) -> dict:
        if len(s) == 0:
            return {}
        return json.loads(s)

    def _calc_score(self, block_id: int, last_accessed: float, cache_hint: dict) -> float:
        with self.profiler.time_operation("calc_score"):
            if cache_hint.get('use_lru'):
                return last_accessed
            if cache_hint.get('use_fifo'):
                return self.id_to_first_access.get(block_id, last_accessed)
            if 'next_timestamp' in cache_hint:
                return -cache_hint['next_timestamp']
            
            prob_has_next = cache_hint.get('prob_has_next')
            if prob_has_next is not None:
                # Use largest seen last_accessed instead of time.time()
                current_time = max(self._max_last_accessed, last_accessed)
                return probability_of_future_arrival(
                    prob_has_next,
                    cache_hint.get('exp_scale', 1.0),
                    current_time - last_accessed
                )
            
            # Default to LRU if no other hint is available
            return last_accessed

    def evict(self) -> Tuple[int, int]:
        with self._rescore_lock, self.profiler.time_operation("evict_total"):
            if not self.free_table:
                raise ValueError("No usable cache memory left")
            
            block_id_to_evict = -1
            content_hash = -1

            # Prioritize evicting outdated blocks
            while self.to_delete_blocks:
                block_id, content_hash, last_accessed = self.to_delete_blocks.pop(0)
                if block_id in self.free_table and self.free_table[block_id].last_accessed == last_accessed:
                    block_id_to_evict = block_id
                    break
            
            # If no outdated block was found/valid, evict based on score
            if block_id_to_evict == -1:
                while self.sorted_dict:
                    _, (block_id, ch) = self.sorted_dict.popitem(0)
                    if block_id in self.free_table:
                        block_id_to_evict = block_id
                        content_hash = ch
                        break
            
            if block_id_to_evict != -1:
                survival_time = time.time() - self.free_table[block_id_to_evict].last_accessed
                self.stat.append("survival_times", survival_time)
                if content_hash == -1:
                    content_hash = self.free_table[block_id_to_evict].content_hash
                
                del self.free_table[block_id_to_evict]
                if block_id_to_evict in self.id_to_first_access:
                    del self.id_to_first_access[block_id_to_evict]
                return block_id_to_evict, content_hash
            
            raise ValueError("Eviction failed: could not find a block to evict.")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        with self._rescore_lock, self.profiler.time_operation("add"):
            # Update maximum seen last_accessed
            self._max_last_accessed = max(self._max_last_accessed, last_accessed)
            
            score = self._calc_score(block_id, last_accessed, cache_hint)
            self.free_table[block_id] = BlockMetaData(content_hash,
                                                      num_hashed_tokens,
                                                      last_accessed,
                                                      cache_hint,
                                                      score)
            self.sorted_dict[(score, last_accessed, block_id)] = (block_id, content_hash)
            if 'id' in cache_hint:
                self.id_to_last_access[cache_hint['id']] = last_accessed
            if block_id not in self.id_to_first_access:
                self.id_to_first_access[block_id] = last_accessed
                
            # Background threading: trigger async refresh
            if time.time() - self.last_refresh_time > self.INSPECT_INTERVAL:
                self._trigger_background_refresh()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        with self._rescore_lock, self.profiler.time_operation("update"):
            if block_id not in self.free_table:
                return

            # Update maximum seen last_accessed
            self._max_last_accessed = max(self._max_last_accessed, last_accessed)

            old_meta = self.free_table[block_id]
            old_entry = (old_meta.score, old_meta.last_accessed, block_id)
            
            if old_entry in self.sorted_dict:
                del self.sorted_dict[old_entry]
            
            score = self._calc_score(block_id, last_accessed, cache_hint)
            self.free_table[block_id].last_accessed = last_accessed
            self.free_table[block_id].cache_hint = cache_hint
            self.free_table[block_id].score = score
            
            self.sorted_dict[(score, last_accessed, block_id)] = (block_id, self.free_table[block_id].content_hash)
            if 'id' in cache_hint:
                self.id_to_last_access[cache_hint['id']] = last_accessed

    def remove(self, block_id: int):
        with self._rescore_lock, self.profiler.time_operation("remove"):
            if block_id not in self.free_table:
                # This can happen due to race conditions or hash collisions
                # where the block was already removed by another thread
                return
            
            old_meta = self.free_table[block_id]
            old_entry = (old_meta.score, old_meta.last_accessed, block_id)
            if old_entry in self.sorted_dict:
                del self.sorted_dict[old_entry]
            del self.free_table[block_id]

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def _trigger_background_refresh(self):
        """Background threading: trigger refresh in background thread"""
        if self._rescore_thread is not None and self._rescore_thread.is_alive():
            return  # Already running
            
        self._rescore_thread = threading.Thread(target=self._background_refresh_worker, daemon=True)
        self._rescore_thread.start()

    def _background_refresh_worker(self):
        """Background worker for rescoring and finding outdated blocks"""
        try:
            with self._rescore_lock:
                with self.profiler.time_operation("background_refresh_total"):
                    self._rescore_all_blocks()
                    self._find_and_mark_outdated_blocks()
                    self.last_refresh_time = time.time()
                    self.refresh_count += 1
                    # Print profiler stats periodically
                    if self.refresh_count % 10 == 9:
                        self.profiler.print_stats()
        except Exception as e:
            print(f"Error in background refresh: {e}")

    def _rescore_all_blocks(self):
        """Rescore all blocks - runs in background thread"""
        with self.profiler.time_operation("rescore_all_blocks"):
            snapshot_items = list(self.free_table.items())  # Snapshot to avoid size change during iteration
            new_sorted_dict = SortedDict()
            for block_id, block_meta in snapshot_items:
                block_meta.score = self._calc_score(block_id, block_meta.last_accessed, block_meta.cache_hint)
                new_sorted_dict[(block_meta.score, block_meta.last_accessed, block_id)] = (block_id, block_meta.content_hash)
            self.sorted_dict = new_sorted_dict
    
    def _find_and_mark_outdated_blocks(self):
        """Efficient outdated block detection: group by conversation ID"""
        with self.profiler.time_operation("find_outdated_blocks"):
            self.to_delete_blocks.clear()
            
            snapshot_items = list(self.free_table.items())  # snapshot
            # Group blocks by conversation ID for efficient processing
            conv_blocks = defaultdict(list)
            for block_id, block_meta in snapshot_items:
                conv_id = block_meta.cache_hint.get('id')
                if conv_id:
                    conv_blocks[conv_id].append((block_id, block_meta))
            
            # Check each conversation's blocks
            for conv_id, blocks in conv_blocks.items():
                latest_access = self.id_to_last_access.get(conv_id, 0)
                for block_id, block_meta in blocks:
                    if latest_access > block_meta.last_accessed:
                        self.to_delete_blocks.append((block_id, block_meta.content_hash, block_meta.last_accessed))

class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self, fifo_mode: bool = False):
        self.fifo_mode = fifo_mode
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.detached_last_accessed: Dict[int, float] = {}
        self.stat = CacheStat()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                survival_time = time.time() - self.free_table[block_id].last_accessed
                self.stat.append("survival_times", survival_time)
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        if self.fifo_mode and block_id in self.detached_last_accessed:
            last_accessed = self.detached_last_accessed.pop(block_id)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if not self.fifo_mode:
            self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        if self.fifo_mode:
            self.detached_last_accessed[block_id] = \
                self.free_table[block_id].last_accessed
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class S3FIFOEvictor(Evictor):
    """S3FIFO eviction with small/main/ghost FIFO queues.

    This is a size-partitioned FIFO with a ghost list for main-queue evictions.
    """

    def __init__(self, config: str):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.queue_by_id: Dict[int, str] = {}
        self.freq_by_id: Dict[int, int] = {}
        self.detached_by_id: Dict[int, Tuple[str, int]] = {}
        self.small_queue = deque()
        self.main_queue = deque()
        self.ghost_queue = deque()
        self.ghost_set = set()
        self.small_size = 0
        self.main_size = 0
        self.stat = CacheStat()
        self.config = self._parse_str_to_dict(config)
        self.small_ratio = self.config.get("small_ratio", 0.1)
        self.ghost_ratio = self.config.get("ghost_ratio", 1)
        self.max_freq = max(1, int(self.config.get("max_freq", 1)))
        print(f"S3FIFOEvictor config: {self.config}")

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def _parse_str_to_dict(self, s: str) -> dict:
        if len(s) == 0:
            return {}
        return json.loads(s)

    def _target_sizes(self) -> Tuple[int, int, int]:
        total = len(self.free_table)
        if total == 0:
            return 0, 0, 0
        small_target = max(1, int(total * self.small_ratio))
        main_target = max(0, total - small_target)
        ghost_target = max(1, int(total * self.ghost_ratio))
        return small_target, main_target, ghost_target

    def _pop_valid(self, queue: deque, queue_name: str) -> int:
        while queue:
            block_id = queue.popleft()
            if block_id not in self.free_table:
                continue
            if self.queue_by_id.get(block_id) != queue_name:
                continue
            return block_id
        return -1

    def _pop_valid_ghost(self) -> int:
        while self.ghost_queue:
            content_hash = self.ghost_queue.popleft()
            if content_hash in self.ghost_set:
                return content_hash
        return -1

    def _trim_ghost(self) -> None:
        _, _, ghost_target = self._target_sizes()
        while len(self.ghost_set) > ghost_target:
            content_hash = self._pop_valid_ghost()
            if content_hash == -1:
                break
            self.ghost_set.discard(content_hash)

    def _add_to_small(self, block_id: int) -> None:
        self.small_queue.append(block_id)
        self.queue_by_id[block_id] = "small"
        self.small_size += 1

    def _add_to_main(self, block_id: int) -> None:
        self.main_queue.append(block_id)
        self.queue_by_id[block_id] = "main"
        self.main_size += 1

    def _add_to_ghost(self, content_hash: int) -> None:
        if content_hash in self.ghost_set:
            return
        self.ghost_set.add(content_hash)
        self.ghost_queue.append(content_hash)
        self._trim_ghost()

    def _remove_from_ghost(self, content_hash: int) -> None:
        if content_hash in self.ghost_set:
            self.ghost_set.discard(content_hash)

    def _evict_block(self,
                     block_id: int,
                     add_to_ghost: bool) -> Tuple[int, int]:
        block_meta = self.free_table.pop(block_id)
        queue_name = self.queue_by_id.pop(block_id, None)
        self.freq_by_id.pop(block_id, None)
        if queue_name == "small":
            self.small_size -= 1
        elif queue_name == "main":
            self.main_size -= 1
        if add_to_ghost:
            self._add_to_ghost(block_meta.content_hash)
        survival_time = time.time() - block_meta.last_accessed
        self.stat.append("survival_times", survival_time)
        return block_id, block_meta.content_hash

    def _evict_from_small(self) -> Tuple[int, int]:
        while True:
            block_id = self._pop_valid(self.small_queue, "small")
            if block_id == -1:
                return -1, -1
            
            freq = self.freq_by_id.get(block_id, 0)
            if freq > 0:
                self.small_size -= 1
                self.freq_by_id[block_id] = 0 
                self._add_to_main(block_id)
                continue 
            else:
                return self._evict_block(block_id, add_to_ghost=True)

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while True:
            small_target, _, _ = self._target_sizes()
            
            if self.small_size > small_target:
                block_id, content_hash = self._evict_from_small()
                if block_id != -1:
                    return block_id, content_hash
            
            block_id, content_hash = self._evict_from_main()
            if block_id != -1:
                return block_id, content_hash
            
            block_id, content_hash = self._evict_from_small()
            if block_id != -1:
                return block_id, content_hash
            
            raise ValueError("Eviction failed: State seems inconsistent.")

    def _evict_from_main(self) -> Tuple[int, int]:
        while True:
            block_id = self._pop_valid(self.main_queue, "main")
            if block_id == -1:
                return -1, -1
            freq = self.freq_by_id.get(block_id, 0)
            if freq > 0:
                self.freq_by_id[block_id] = freq - 1
                self.main_queue.append(block_id)
                continue
            return self._evict_block(block_id, add_to_ghost=True)

    def add(self,
            block_id: int,
            content_hash: int,
            num_hashed_tokens: int,
            last_accessed: float,
            cache_hint: dict):
        self.free_table[block_id] = BlockMetaData(
            content_hash=content_hash,
            num_hashed_tokens=num_hashed_tokens,
            last_accessed=last_accessed,
            cache_hint=cache_hint,
        )
        if block_id in self.detached_by_id:
            queue_name, freq = self.detached_by_id.pop(block_id)
            self._remove_from_ghost(content_hash)
            if queue_name == "main":
                self._add_to_main(block_id)
            else:
                self._add_to_small(block_id)
            self.freq_by_id[block_id] = min(self.max_freq, max(0, freq))
            return

        if content_hash in self.ghost_set:
            self._remove_from_ghost(content_hash)
            self._add_to_main(block_id)
            self.freq_by_id[block_id] = 1
        else:
            self._add_to_small(block_id)
            self.freq_by_id[block_id] = 0

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if block_id not in self.free_table:
            return
        block_meta = self.free_table[block_id]
        block_meta.last_accessed = last_accessed
        block_meta.cache_hint = cache_hint
        freq = self.freq_by_id.get(block_id, 0)
        self.freq_by_id[block_id] = min(self.max_freq, freq + 1)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            return
        queue_name = self.queue_by_id.pop(block_id, None)
        freq = self.freq_by_id.pop(block_id, 0)
        self.free_table.pop(block_id)
        # Preserve original queue and bump freq
        self.detached_by_id[block_id] = (queue_name or "small",
                                         min(self.max_freq, freq + 1))
        if queue_name == "small":
            self.small_size -= 1
        elif queue_name == "main":
            self.main_size -= 1

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

def make_evictor(eviction_algorithm: str, config: str) -> Evictor:
    if eviction_algorithm == 'lru':
        return LRUEvictor(fifo_mode=False)
    if eviction_algorithm == 'fifo':
        return LRUEvictor(fifo_mode=True)
    if 's3fifo' in eviction_algorithm:
        return S3FIFOEvictor(config)
    return LRUMLEvictor(config)
