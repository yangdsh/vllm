# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
import time
import statistics
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


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
        self.log_interval = 5  # Log every 5 seconds
    
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

        for key in self.stat.keys():
            data = self.stat[key]
            if not data:
                return 0
            
            mean = statistics.mean(data)
            std_dev = statistics.stdev(data) if len(data) > 1 else 0
            print(f"Summary for {key}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")
        return 1
        

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

class LRUMLEvictor:

    def __init__(self, config):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.queue_map: Dict[int, Tuple[float, int, int, int]] = {}  # Maps block_id to priority queue entry
        self.config = self.parse_str_to_dict(config)
        self.last_clean_time = time.time()

    def parse_str_to_dict(self, s: str) -> dict:
        if len(s) == 0:
            return {}
        return {key: value for key, value in (pair.split("=", 1) for pair in s.split(","))}

    def probability_of_future_arrival(self, prob_has_next, exp_scale, elapsed_time, future_window):
        if prob_has_next == 0:
            return 0.0
        prob_survived_elapsed = np.exp(-elapsed_time / exp_scale)
        prob_not_survive_future = 1 - np.exp(-future_window / exp_scale)
        return (prob_has_next * prob_survived_elapsed * prob_not_survive_future) / (
                prob_has_next * prob_survived_elapsed + (1 - prob_has_next)
        )

    def calc_score(self, last_accessed, cache_hint):
        if 'next_timestamp' in cache_hint:
            return -cache_hint['next_timestamp']
        if 'prob_has_next' in cache_hint:
            return self.probability_of_future_arrival(
                cache_hint['prob_has_next'], cache_hint['exp_scale'],
                time.time() - last_accessed, 10)
        if cache_hint['turns'] <= 6:
            return 6 - cache_hint['turns'] + last_accessed
        else:
            return max(cache_hint['turns'] - 6, 10) + last_accessed

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")
        
        while self.priority_queue:
            score, _, block_id, content_hash = heapq.heappop(self.priority_queue)
            if block_id in self.free_table and self.free_table[block_id].score == score:
                del self.queue_map[block_id]  # Remove from tracking
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        score = self.calc_score(last_accessed, cache_hint)
        metadata = BlockMetaData(content_hash, num_hashed_tokens, last_accessed, cache_hint, score)
        self.free_table[block_id] = metadata
        entry = (score, -num_hashed_tokens, block_id, content_hash)
        heapq.heappush(self.priority_queue, entry)
        self.queue_map[block_id] = entry  # Track position in queue

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if block_id not in self.free_table:
            raise ValueError("Block ID not found in free table")
        
        self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint
        self.free_table[block_id].score = self.calc_score(last_accessed, cache_hint)
        
        # Remove outdated entry and insert updated one
        if block_id in self.queue_map:
            old_entry = self.queue_map.pop(block_id)
            self.priority_queue.remove(old_entry)
            heapq.heapify(self.priority_queue)
        
        new_entry = (self.free_table[block_id].score, -self.free_table[block_id].num_hashed_tokens, block_id, self.free_table[block_id].content_hash)
        heapq.heappush(self.priority_queue, new_entry)
        self.queue_map[block_id] = new_entry

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        
        if block_id in self.queue_map:
            old_entry = self.queue_map.pop(block_id)
            self.priority_queue.remove(old_entry)
            heapq.heapify(self.priority_queue)
        
        del self.free_table[block_id]

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

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

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
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
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
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
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def make_evictor(eviction_algorithm: str, config: str) -> Evictor:
    if eviction_algorithm == 'lru':
        return LRUEvictor()
    elif eviction_algorithm == 'lru-ml':
        return LRUMLEvictor(config)
    elif eviction_algorithm == 'lruml':
        return LRUMLEvictor(config)
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_algorithm}")
