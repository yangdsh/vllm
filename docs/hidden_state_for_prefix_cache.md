# Hidden State Embedding Feature for ML-Based Eviction

## Overview

This document describes the feature that enables using hidden states from the transformer's last layer as input embeddings for the ML-based eviction model. Instead of using text embeddings from a SentenceTransformer, the model can now use the actual hidden state representation from the LLM.

## Motivation

- **Efficiency**: Extracting hidden states from existing model output is more efficient than encoding text with a separate model
- **Relevance**: Hidden states directly represent the model's internal understanding of the sequence
- **Quality**: Uses the same representation the model uses for token generation
- **Simplicity**: Hidden states are a single tensor, simpler than KV cache extraction

## Architecture

### Components Modified

1. **`evictor_ml_model.py`** - ML model updated to accept hidden states directly
2. **`evictor_ml_manager.py`** - Batch prediction path updated to use hidden states
3. **`block_manager.py`** - Block manager extracts hidden states from model output

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Model Runner  │────▶│  Scheduler/      │────▶│ OnlineLearningMgr   │
│ (returns        │     │  LLM Engine      │     │ (extract_hidden_    │
│  hidden_states) │     │ (process output) │     │  stats & training)  │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │ block_manager.      │
                        │ extract_hidden_     │
                        │ states_for_prefix_  │
                        │ cache()             │
                        └─────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────┐
                        │  cache_hint dict    │
                        │ (hidden_state added)│
                        └─────────────────────┘
```

## Code Changes

### 1. `MLModel` Class Updates (evictor_ml_model.py)

**New Constructor Parameters:**
- `use_hidden_state_embeddings`: If True, use hidden states instead of text embeddings
- `hidden_state_dim`: Dimension of hidden states (required if use_hidden_state_embeddings=True)

**Updated Methods:**
- `predict_single_processed()`: Added `hidden_state` parameter
- `predict_batch_processed()`: Added `hidden_states_list` parameter

Both methods now support:
- Text embeddings (original behavior)
- Hidden state embeddings (new)
- Dimension mismatch handling (fall back to zero embedding if size is wrong)

### 2. `OnlineLearningManager` Updates (evictor_ml_manager.py)

**Updated `_process_prediction_batch()`:**
- Checks for `hidden_state` in each `cache_hint`
- If every item in the batch has a hidden state, calls  
  `ml_model.predict_batch_processed(hidden_states_list=..., turns=...)`
- Otherwise falls back to text embeddings via `predict_batch_processed(texts, turns)`

### 3. `SelfAttnBlockSpaceManager` Updates (block_manager.py)

**New Method:**
- `extract_hidden_states_for_prefix_cache()`: Extracts hidden states for all
  running sequences in a `SequenceGroup` and stores them in `seq.cache_hint["hidden_state"]`

## Usage

### Automatic Usage (Default)

Hidden state extraction happens when:
1. The eviction algorithm contains "ml" (enables OnlineLearningManager)
2. `SamplerOutput.hidden_states` is available (one vector per prompt)
3. `extract_hidden_states_for_prefix_cache()` is called after model execution

The extracted hidden states are stored in `seq.cache_hint["hidden_state"]`.

### Manual Usage

```python
# Hidden states from sampler output: one vector per prompt (last token)
hidden_states = sampler_output.hidden_states  # [num_prompts, hidden_size]
hidden_state = hidden_states[0]  # First prompt in batch
print(f"Hidden state shape: {hidden_state.shape}")
```

### Using Hidden State Embeddings in ML Model

```python
from vllm.core.evictor_ml_model import MLModel

# Initialize model with hidden state embedding support
model = MLModel(
    task="classification",
    use_hidden_state_embeddings=True,
    hidden_state_dim=4096,  # e.g., model's hidden size
    train_mode=True
)

# Predict with hidden state
prob = model.predict_single_processed(
    turns=5,
    hidden_state=hidden_state
)

# Batch prediction
probs = model.predict_batch_processed(
    hidden_states_list=[h1, h2, ...],
    turns=[5, 3, ...]
)
```

### Integration Point

Hidden state extraction is automatically called in `llm_engine.py`'s 
`_process_model_outputs()` method when processing prefill outputs:

```python
if (outputs and isinstance(outputs[0], SamplerOutput)
        and outputs[0].hidden_states is not None):
    hidden_states = outputs[0].hidden_states  # [num_prompts, hidden_size]

    seq_idx_offset = 0
    for sgm, scheduled_sg in zip(seq_group_metadata_list,
                                 scheduler_outputs.scheduled_seq_groups):
        if not sgm.is_prompt:
            continue

        scheduler = self.scheduler[0]
        block_manager = getattr(scheduler, "block_manager", None)
        if hasattr(block_manager, "extract_hidden_states_for_prefix_cache"):
            block_manager.extract_hidden_states_for_prefix_cache(
                seq_group=scheduled_sg.seq_group,
                hidden_states=hidden_states,
                seq_idx_offset=seq_idx_offset,
            )
        seq_idx_offset += 1
```

## Logging

Key log messages (at INFO/DEBUG level):
- `"Hidden state extracted for seq_id=X, shape=Y"` - Successful extraction
- `"Error extracting hidden state for seq X: Y"` - Extraction error

## Configuration

No additional configuration is required. The feature is enabled automatically when 
using ML-based eviction algorithms.

### Eviction Algorithm Options

Prefix cache eviction supports `lru` and `s3fifo`. For S3FIFO, optional config
keys are available in `eviction_algorithm_config`:

```json
{
    "small_ratio": 0.1,
    "ghost_ratio": 0.1,
    "max_freq": 3
}
```

### Data Collection for Offline Training

To collect training data (hidden states + labels) for offline MLP training:

```python
# In eviction_algorithm_config JSON:
{
    "enable_data_collection": true,
    "data_collection_dir": "training_data"
}
```

Or via command line:
```bash
--eviction_algorithm_config '{"enable_data_collection": true, "data_collection_dir": "/path/to/output"}'
```

This will create:
- `training_data/hidden_states.npy` - NumPy array of shape `[N, hidden_size]`
- `training_data/labels.jsonl` - JSONL file with metadata for each sample

Each line in `labels.jsonl` contains:
```json
{"conv_id": "123", "turns": 2, "label": 1, "text": "...", "hidden_state_shape": [4096]}
```

Where `label=1` means the conversation had a follow-up turn, `label=0` means it timed out.

## Comparison with Text Embeddings

| Aspect | Text Embeddings | Hidden State |
|--------|----------------|--------------|
| Source | SentenceTransformer | LLM's last layer |
| Dimension | ~384 (e.g., MiniLM) | ~4096 (model dependent) |
| Compute | Separate encoding | Already computed |
| Relevance | General text similarity | Model's actual understanding |

## Limitations

1. **Dimension**: Hidden states are typically larger than text embeddings
2. **Availability**: Only available after prefill (predictions during wait use text)

