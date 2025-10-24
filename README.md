# RAG-Fuse

A Retrieval-Augmented Generation pipeline for multi-class text classification (MCTC) using hybrid sparse and dense retrieval with LLM-enhanced label descriptions.

## Overview

RAG-Fuse combines sparse (BM25) and dense (BERT/RoBERTa) retrieval methods with LLM-generated label descriptions to improve MCTC effectiveness and efficiency. The system transforms multi-class classification into a ranking problem, using a multi-stage pipeline involving retrieval, fusion, and aggregation to predict relevant labels for text documents.

## Key Features

- **Hybrid Retrieval**: Seamlessly combines BM25 sparse retrieval with dense neural retrievers
- **LLM Label Enhancement**: Leverages LLaMA models to generate enriched label descriptions (RAG-labels)
- **Intelligent Ranking Fusion**: Merges multiple ranking strategies using z-score normalization and MNZ fusion
- **Propensity-Scored Metrics**: Comprehensive evaluation with propensity-scored precision and nDCG
- **Multi-Dataset Support**: Pre-configured for ACM, OHSUMED, REUTERS, and TWITTER datasets
- **Head/Tail Label Handling**: Separate optimization for frequent and rare labels

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090 or RTX 4090 (recommended for optimal performance)
- **CUDA**: Version 12.8.1 or higher
- **RAM**: 32GB+ recommended for large datasets
- **OS**: Ubuntu 22.04 LTS

### Software Dependencies
- Python 3.10
- Conda package manager
- NLTK data packages (punkt, punkt_tab)
- PyTorch 2.8.0+
- PyTorch Lightning 2.5.3

## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:celsofranssa/RAG-Fuse.git
cd RAG-Fuse
git checkout SBBD2025
```

### 2. Set Up Python Environment
```bash
# Create conda environment
conda create -n RAG-Fuse python=3.10 -y

# Activate environment
conda activate RAG-Fuse

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python3 -m nltk.downloader punkt
python3 -m nltk.downloader punkt_tab
```

## Project Structure

```
RAG-Fuse/
├── main.py                 # Main entry point for all tasks
├── run.sh                  # Master execution script
├── requirements.txt        # Python dependencies
│
├── run/                    # Dataset-specific execution scripts
│   ├── ACM.sh             # ACM dataset pipeline
│   ├── OHSUMED.sh         # OHSUMED dataset pipeline
│   ├── REUTERS.sh         # REUTERS dataset pipeline
│   └── TWITTER.sh         # TWITTER dataset pipeline
│
├── setting/               # Configuration files (Hydra)
│   ├── setting.yaml       # Main configuration
│   ├── data/              # Dataset-specific configs
│   │   ├── ACM.yaml
│   │   ├── OHSUMED.yaml
│   │   ├── REUTERS.yaml
│   │   └── TWITTER.yaml
│   └── model/             # Model configurations
│       ├── BM25.yaml
│       ├── RetrieverBERT.yaml
│       └── RetrieverRoBERTa.yaml
│
├── source/                # Source code modules
│   ├── callback/          # PyTorch Lightning callbacks
│   ├── datamodule/        # Data loading and processing
│   ├── dataset/           # Dataset classes
│   ├── distance/          # Distance/similarity metrics
│   ├── encoder/           # Neural encoder architectures
│   ├── helper/            # Pipeline helper classes
│   ├── loss/              # Loss functions (NTXent, etc.)
│   ├── metric/            # Evaluation metrics
│   ├── miner/             # Hard negative mining
│   ├── model/             # Model definitions
│   └── pooling/           # Pooling strategies
│
└── resource/              # Runtime resources (auto-generated)
    ├── dataset/           # Dataset files (*.pkl)
    ├── log/               # TensorBoard training logs
    ├── model_checkpoint/  # Saved model weights
    ├── ranking/           # Generated ranking files
    ├── prediction/        # Model predictions
    ├── result/            # Evaluation results (*.rts)
    └── llm/               # LLM prompts and responses
```

## Quick Start

### Basic Usage

Run the complete pipeline on a single fold:
```bash
# Run on ACM dataset, fold 0
bash run.sh ACM 0 0
```

Run on multiple folds:
```bash
# Run on folds 0 through 4
bash run.sh ACM 0 4

# Run complete 10-fold cross-validation
bash run.sh OHSUMED 0 9
bash run.sh REUTERS 0 9
bash run.sh TWITTER 0 9
```

### Command Syntax
```bash
bash run.sh <DATASET> <START_FOLD_IDX> <END_FOLD_IDX>
```

**Parameters:**
- `<DATASET>`: Dataset name (ACM, OHSUMED, REUTERS, TWITTER)
- `<START_FOLD_IDX>`: Starting fold index (0-9)
- `<END_FOLD_IDX>`: Ending fold index (inclusive, 0-9)

## Pipeline Stages

The RAG-Fuse pipeline consists of eight sequential stages:

### Stage 1: Sparse Retrieval (BM25)
Performs initial retrieval using the BM25 algorithm on text documents. Creates baseline rankings for both head and tail labels.

**Output:** `resource/ranking/BM25_<DATASET>/`

### Stage 2: Prompt Optimization (Optional)
Optimizes the prompt template used for generating RAG-labels through iterative refinement based on similarity metrics.

**Command:**
```bash
python main.py tasks=[prompt_opt] data=ACM
```

**Output:** `resource/llm/<DATASET>/optimized_prompt.txt`

### Stage 3: Label Description Generation (Optional)
Uses LLMs (LLaMA) to generate enriched, contextualized descriptions for each label based on training examples.

**Command:**
```bash
python main.py tasks=[label_desc] data=ACM data.folds=[0]
```

**Output:** `resource/dataset/<DATASET>/fold_<N>/labels_descriptions.pkl`

### Stage 4: Dense Retrieval Training
Fine-tunes BERT/RoBERTa encoders using contrastive learning (NTXent loss) to map texts and labels into a shared embedding space.

**Configuration:**
- Max epochs: 5
- Early stopping patience: 3
- Loss: NTXent with temperature 0.07
- Optimizer: AdamW with linear warmup

**Output:** `resource/model_checkpoint/`

### Stage 5: Dense Retrieval Prediction
Generates embeddings for all texts and labels using the trained dense retriever.

**Output:** `resource/prediction/<MODEL_NAME>_<DATASET>/fold_<N>/`

### Stage 6: Dense Retrieval Evaluation
Evaluates the dense retriever using HNSW (Hierarchical Navigable Small World) indexing for efficient approximate nearest neighbor search.

**Metrics:** Precision@k, nDCG@k, MRR

**Output:** `resource/result/<MODEL_NAME>_<DATASET>/`

### Stage 7: Ranking Fusion
Combines sparse (BM25) and dense retrieval rankings using:
- **Normalization:** Z-score (zmuv)
- **Fusion Method:** MNZ (Maximum Normalized Score)

**Output:** `resource/ranking/Fused_<MODEL_NAME>_<DATASET>/`

### Stage 8: Ranking Aggregation
Aggregates head and tail label rankings into final unified rankings. Computes comprehensive metrics including propensity-scored variants.

**Output:** `resource/ranking/Aggregated_<MODEL_NAME>_<DATASET>/`

## Configuration

### Dataset Configuration

Edit files in `setting/data/` to customize dataset-specific parameters:

**Example: `ACM.yaml`**
```yaml
name: ACM
dir: resource/dataset/ACM/
folds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
text_max_length: 256
label_max_length: 256
batch_size: 64
num_labels: 11
label_enhancement: LLM  # Options: RAW, PMI, LLM
```

**Key Parameters:**
- `text_max_length`: Maximum sequence length for text tokenization
- `label_max_length`: Maximum sequence length for label tokenization
- `batch_size`: Training and inference batch size
- `num_labels`: Total number of labels in dataset
- `label_enhancement`: Strategy for label enrichment
  - `RAW`: Use raw label text only
  - `PMI`: Use PMI-based pseudo-labels
  - `LLM`: Use LLM-generated descriptions

### Model Configuration

Edit files in `setting/model/` to configure model architectures:

**Example: `RetrieverBERT.yaml`**
```yaml
name: RetrieverBERT
type: retriever
encoder:
  architecture: bert-base-uncased
  pooling: ConcatenatePooling  # Last 4 hidden layers
hidden_size: 3072  # 4 layers × 768 dims
lr: 5e-5
loss:
  criterion:
    temperature: 0.07
```

**Available Models:**
- `RetrieverBERT`: BERT-based dense retriever
- `RetrieverRoBERTa`: RoBERTa-based dense retriever
- `BM25`: Sparse retriever baseline

### Training Configuration

Edit `setting/setting.yaml` to control training behavior:

```yaml
trainer:
  max_epochs: 5
  patience: 3              # Early stopping patience
  devices: [0]             # GPU device IDs
  precision: 16-mixed      # Mixed precision training

eval:
  metrics: [ndcg, precision]
  thresholds: [1, 5, 10]
  num_nearest_neighbors: 100
```

## Evaluation Metrics

### Traditional Ranking Metrics
- **Precision@k**: Proportion of relevant labels in top-k predictions
- **nDCG@k**: Normalized Discounted Cumulative Gain at cutoff k
- **MRR**: Mean Reciprocal Rank of the first relevant label

### Propensity-Scored Metrics
Accounts for label frequency bias in extreme classification:
- **PSPrecision@k**: Propensity-scored precision
- **PSnDCG@k**: Propensity-scored nDCG

Propensity scores computed as: `1 + C × (freq + B)^(-A)`

### Label-Specific Evaluation
Metrics are computed separately for:
- **Head labels**: Frequent labels (e.g., frequency > threshold)
- **Tail labels**: Rare labels (e.g., frequency ≤ threshold)

This allows fine-grained analysis of model performance across the label frequency spectrum.

## Dataset Format

### Required Directory Structure
```
resource/dataset/<DATASET_NAME>/
├── samples.pkl              # All samples (list of dicts)
├── relevance_map.pkl        # Ground truth: {text_idx: [label_ids]}
├── label_cls.pkl            # Label classification: {label_idx: ['head'/'tail']}
├── text_cls.pkl             # Text classification: {text_idx: ['head'/'tail']}
├── propensities.pkl         # Propensity scores per label
└── fold_<N>/                # Per-fold splits (N = 0-9)
    ├── train.pkl            # Training sample IDs
    ├── val.pkl              # Validation sample IDs
    ├── test.pkl             # Test sample IDs
    └── labels_descriptions.pkl  # LLM-generated descriptions (optional)
```

### Sample Format (`samples.pkl`)
```python
[
    {
        "idx": 0,                          # Sample index
        "text_idx": 0,                     # Text identifier
        "text": "sample text...",          # Raw text content
        "labels": ["label1", "label2"],    # Label strings
        "labels_ids": [0, 5],              # Label indices
        "keywords": [("word", 0.9), ...]   # Optional: keywords
    },
    ...
]
```

## Important Notes

### ⚠️ Critical: Clean Resource Directories

Before starting a new experiment, **always clear** these directories:

```bash
# Clear all cached resources
rm -rf resource/log/*
rm -rf resource/model_checkpoint/*
rm -rf resource/prediction/*

# Or selectively clear for specific dataset
rm -rf resource/log/<MODEL_NAME>_<DATASET>_*
rm -rf resource/model_checkpoint/<MODEL_NAME>_<DATASET>_*
```

**Why?** The pipeline does not overwrite existing checkpoints and logs. Stale files will cause:
- Training to resume from old checkpoints
- Predictions using outdated models
- Incorrect evaluation results

### Best Practices

1. **Always use version control** for your configuration files
2. **Document your experiments** by saving the configuration used
3. **Monitor training** using TensorBoard: `tensorboard --logdir resource/log/`
4. **Verify dataset format** before running the pipeline
5. **Use appropriate batch sizes** based on your GPU memory

## Advanced Features

### Task-Specific Execution

Run individual pipeline stages:

```bash
# Only sparse retrieval
python main.py tasks=[sparse_retrieve] data=ACM data.folds=[0]

# Only train dense retriever
python main.py tasks=[fit] data=ACM data.folds=[0]

# Only generate predictions
python main.py tasks=[predict] data=ACM data.folds=[0]

# Multiple tasks in sequence
python main.py tasks=[fit,predict,eval] data=ACM data.folds=[0]
```

### Available Tasks
| Task | Description |
|------|-------------|
| `sparse_retrieve` | BM25 sparse retrieval |
| `fit` | Train dense retriever |
| `predict` | Generate embeddings |
| `eval` | Evaluate retriever |
| `fuse` | Fuse sparse + dense rankings |
| `aggregate` | Aggregate head + tail rankings |
| `prompt_opt` | Optimize LLM prompts |
| `label_desc` | Generate label descriptions |

### LLM Configuration

Configure LLM parameters in `setting/setting.yaml`:

```yaml
llm:
  prompt_opt:
    model: meta.llama3-1-8b-instruct-v1:0
    num_epochs: 3
    temperature: 0.5
    
  label_desc:
    batch_size: 32
    max_gen_len: 256
    temperature: 0.6
    num_samples: 5  # Samples per label for description
```

## Results and Outputs

### Output Locations

| Resource Type | Path | Format |
|--------------|------|--------|
| Rankings | `resource/ranking/<MODEL>_<DATASET>/` | `.rnk` (pickle) |
| Metrics | `resource/result/<MODEL>_<DATASET>/` | `.rts` (TSV) |
| Predictions | `resource/prediction/<MODEL>_<DATASET>/` | `.prd` (pickle) |
| Checkpoints | `resource/model_checkpoint/` | `.ckpt` (PyTorch) |
| Logs | `resource/log/` | TensorBoard format |

### Results Format

Results files (`.rts`) are tab-separated with columns:
```
fold_idx    split    cls     ndcg@1    ndcg@5    precision@1    precision@5    psnDCG@1    psnDCG@5    ...
0          test     head    0.856     0.923     0.850          0.920          0.801       0.885       ...
0          test     tail    0.723     0.812     0.715          0.805          0.682       0.768       ...
```

### Visualizing Results

Use TensorBoard to monitor training:
```bash
tensorboard --logdir resource/log/ --port 6006
```

Analyze results with pandas:
```python
import pandas as pd
results = pd.read_csv('resource/result/LLM_RetrieverBERT_ACM/...rts', sep='\t')
print(results.groupby('cls').mean())
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **CUDA out of memory** | Batch size too large | Reduce `batch_size` in data config |
| **Old checkpoints interfere** | Stale cached files | Clear `resource/log/` and `resource/model_checkpoint/` |
| **NLTK data not found** | Missing NLTK packages | Run `python3 -m nltk.downloader punkt punkt_tab` |
| **nmslib installation fails** | Binary compatibility | Install from GitHub: `pip install git+https://github.com/nmslib/nmslib/` |
| **Slow training** | CPU bottleneck | Increase `num_workers` in data config |
| **NaN loss values** | Learning rate too high | Reduce `lr` in model config |
| **Poor tail performance** | Insufficient tail samples | Adjust label classification threshold |

### Debug Mode

Enable debug mode for detailed logging:
```bash
python main.py tasks=[fit] data=ACM trainer.fast_dev_run=True
```

### Logging Configuration

Set logging level in your script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Use mixed precision training** (`precision: 16-mixed`) for 2-3x speedup
2. **Optimize batch size** to maximize GPU utilization
3. **Enable persistent workers** (`persistent_workers=True` in DataLoader)
4. **Use SSD storage** for faster data loading
5. **Profile your code** to identify bottlenecks:
   ```bash
   python -m cProfile -o profile.stats main.py tasks=[fit] data=ACM
   ```

## Citation

If you use RAG-Fuse in your research, please cite:

```bibtex
@inproceedings{Franca2025RAGFuse,
  title={Muitas Classes Desbalanceadas? Não Classifique-Ranqueie! Uma Abordagem Baseada em Retrieval-Augmented Generation (RAG)-labels para Classificação Textual Multi-classe},
  author={França, Celso and Nunes, Ian and Salles, Thiago and Cunha, Washington and Jallais, Gabriel and Rocha, Leonardo and Gonçalves, Marcos André},
  booktitle={Simpósio Brasileiro de Banco de Dados (SBBD)},
  pages={264--277},
  year={2025},
  organization={SBC}
}
```

## Acknowledgments

- Developed by Celso França and collaborators
- Built with PyTorch, PyTorch Lightning, and Transformers
- Uses Hydra for configuration management
- Sparse retrieval powered by retriv
- Dense retrieval indexing via nmslib

## Contact

For questions, issues, or collaboration opportunities:
- **GitHub Issues**: [github.com/celsofranssa/RAG-Fuse/issues](https://github.com/celsofranssa/RAG-Fuse/issues)
- **Primary Author**: Celso França

---

**Note**: This is research code associated with the SBBD 2025 paper. For production use, additional testing and optimization may be required.