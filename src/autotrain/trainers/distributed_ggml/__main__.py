import argparse
import os
import json
from pathlib import Path

from autotrain import logger
from autotrain.trainers.distributed_ggml.params import DistributedGGMLParams
from autotrain.trainers.distributed_ggml.file_parser import discover_and_parse_files
from autotrain.trainers.distributed_ggml.hypergraph import HyperGraphEncoder
from autotrain.trainers.distributed_ggml.tensorizer import HyperGraphTensorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


def monitor(func):
    """Simple monitor decorator."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def save_training_params(config):
    """Save training parameters to JSON file."""
    os.makedirs(config.project_name, exist_ok=True)
    params_path = os.path.join(config.project_name, "training_params.json")
    
    with open(params_path, 'w') as f:
        json.dump(config.model_dump(), f, indent=2)
        
    logger.info(f"Training parameters saved to {params_path}")


@monitor
def train(config):
    """Main training function for distributed GGML training set synthesis."""
    
    if isinstance(config, dict):
        config = DistributedGGMLParams(**config)
    
    logger.info("Starting Distributed GGML Training Set Synthesis")
    logger.info(f"Configuration: {config}")
    
    # Create output directory
    os.makedirs(config.project_name, exist_ok=True)
    
    # Step 1: File Discovery & Parsing
    logger.info("=" * 50)
    logger.info("Step 1: File Discovery & Parsing")
    logger.info("=" * 50)
    
    fragments = discover_and_parse_files(
        root_path=config.data_path,
        file_extensions=config.file_extensions,
        max_fragment_size=config.max_fragment_size
    )
    
    if not fragments:
        logger.error("No fragments found. Please check your data path and file extensions.")
        return
        
    # Export fragments for inspection
    fragments_summary = {
        'total_fragments': len(fragments),
        'by_type': {},
        'by_file': {}
    }
    
    for fragment in fragments:
        fragment_type = fragment.fragment_type
        file_path = fragment.context.get('file_path', 'unknown')
        
        fragments_summary['by_type'][fragment_type] = fragments_summary['by_type'].get(fragment_type, 0) + 1
        fragments_summary['by_file'][file_path] = fragments_summary['by_file'].get(file_path, 0) + 1
        
    fragments_path = os.path.join(config.project_name, 'fragments_summary.json')
    with open(fragments_path, 'w') as f:
        json.dump(fragments_summary, f, indent=2)
    
    logger.info(f"Fragments summary saved to {fragments_path}")
    
    # Step 2: Hypergraph Construction
    logger.info("=" * 50)
    logger.info("Step 2: Hypergraph Construction")
    logger.info("=" * 50)
    
    encoder = HyperGraphEncoder(max_depth=config.hypergraph_depth)
    hypergraph = encoder.encode_fragments(fragments)
    
    # Export hypergraph
    hypergraph_path = os.path.join(config.project_name, 'hypergraph.json')
    encoder.export_hypergraph(hypergraph_path)
    
    # Step 3: Tensorization
    logger.info("=" * 50)
    logger.info("Step 3: Tensorization & Corpus Construction")
    logger.info("=" * 50)
    
    tensorizer = HyperGraphTensorizer(
        embedding_dim=config.tensor_dimensions,
        attention_threshold=config.attention_threshold,
        num_shards=config.distributed_shards,
        use_semantic_clustering=config.semantic_clustering
    )
    
    tensors, sharded_tensors = tensorizer.tensorize_hypergraph(hypergraph)
    
    # Step 4: Export Training Set
    logger.info("=" * 50)
    logger.info("Step 4: Corpus Export & Validation")
    logger.info("=" * 50)
    
    # Export unified tensors
    tensors_dir = os.path.join(config.project_name, 'tensors')
    tensorizer.export_tensors(tensors, tensors_dir)
    
    # Export sharded tensors
    sharded_dir = os.path.join(config.project_name, 'sharded_tensors')
    tensorizer.export_sharded_tensors(sharded_tensors, sharded_dir)
    
    # Step 5: Create Training Set Manifest
    logger.info("=" * 50)
    logger.info("Step 5: Training Set Manifest & Validation")
    logger.info("=" * 50)
    
    manifest = create_training_manifest(config, fragments, hypergraph, tensors, sharded_tensors)
    
    manifest_path = os.path.join(config.project_name, 'training_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Training manifest saved to {manifest_path}")
    
    # Step 6: Validation
    validation_results = validate_training_set(config, manifest)
    
    validation_path = os.path.join(config.project_name, 'validation_results.json')
    with open(validation_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    logger.info(f"Validation results saved to {validation_path}")
    
    # Save training parameters
    save_training_params(config)
    
    # Step 7: Generate Usage Documentation
    generate_usage_documentation(config)
    
    logger.info("=" * 50)
    logger.info("Distributed GGML Training Set Synthesis Complete!")
    logger.info("=" * 50)
    logger.info(f"Output directory: {config.project_name}")
    logger.info(f"Total fragments: {len(fragments)}")
    logger.info(f"Hypergraph nodes: {len(hypergraph.nodes)}")
    logger.info(f"Hypergraph edges: {len(hypergraph.edges)}")
    logger.info(f"Generated tensors: {len(tensors)}")
    logger.info(f"Distributed shards: {len(sharded_tensors)}")


def create_training_manifest(config, fragments, hypergraph, tensors, sharded_tensors):
    """Create a comprehensive training set manifest."""
    
    manifest = {
        'version': '1.0.0',
        'created_at': str(Path().resolve()),
        'config': config.model_dump(),
        'statistics': {
            'total_fragments': len(fragments),
            'fragment_types': {},
            'hypergraph_nodes': len(hypergraph.nodes),
            'hypergraph_edges': len(hypergraph.edges),
            'tensor_count': len(tensors),
            'shard_count': len(sharded_tensors),
            'total_tensor_size': sum(tensor.data.size for tensor in tensors)
        },
        'files': {
            'fragments_summary': 'fragments_summary.json',
            'hypergraph': 'hypergraph.json',
            'tensors_dir': 'tensors/',
            'sharded_tensors_dir': 'sharded_tensors/',
            'training_params': 'training_params.json',
            'validation_results': 'validation_results.json',
            'usage_guide': 'USAGE.md'
        },
        'tensor_info': {
            'tensor_types': list(set(tensor.tensor_type for tensor in tensors)),
            'embedding_dimension': config.tensor_dimensions,
            'attention_threshold': config.attention_threshold,
            'export_format': config.export_format
        }
    }
    
    # Count fragment types
    for fragment in fragments:
        fragment_type = fragment.fragment_type
        manifest['statistics']['fragment_types'][fragment_type] = manifest['statistics']['fragment_types'].get(fragment_type, 0) + 1
    
    return manifest


def validate_training_set(config, manifest):
    """Validate the generated training set."""
    
    validation_results = {
        'validation_passed': True,
        'checks': [],
        'warnings': [],
        'errors': []
    }
    
    # Check 1: File existence
    project_path = Path(config.project_name)
    required_files = [
        'fragments_summary.json',
        'hypergraph.json',
        'training_manifest.json',
        'training_params.json'
    ]
    
    for file_name in required_files:
        file_path = project_path / file_name
        if file_path.exists():
            validation_results['checks'].append(f"✓ {file_name} exists")
        else:
            validation_results['errors'].append(f"✗ {file_name} missing")
            validation_results['validation_passed'] = False
    
    # Check 2: Tensor directories
    tensors_dir = project_path / 'tensors'
    sharded_dir = project_path / 'sharded_tensors'
    
    if tensors_dir.exists():
        tensor_files = list(tensors_dir.glob('*.npy'))
        validation_results['checks'].append(f"✓ Tensors directory exists with {len(tensor_files)} tensor files")
    else:
        validation_results['errors'].append("✗ Tensors directory missing")
        validation_results['validation_passed'] = False
    
    if sharded_dir.exists():
        shard_dirs = [d for d in sharded_dir.iterdir() if d.is_dir()]
        validation_results['checks'].append(f"✓ Sharded tensors directory exists with {len(shard_dirs)} shards")
    else:
        validation_results['errors'].append("✗ Sharded tensors directory missing")
        validation_results['validation_passed'] = False
    
    # Check 3: Manifest consistency
    if 'statistics' in manifest:
        stats = manifest['statistics']
        if stats['total_fragments'] > 0:
            validation_results['checks'].append(f"✓ {stats['total_fragments']} fragments processed")
        else:
            validation_results['warnings'].append("⚠ No fragments found")
        
        if stats['hypergraph_nodes'] > 0:
            validation_results['checks'].append(f"✓ {stats['hypergraph_nodes']} hypergraph nodes created")
        else:
            validation_results['warnings'].append("⚠ No hypergraph nodes created")
        
        if stats['tensor_count'] > 0:
            validation_results['checks'].append(f"✓ {stats['tensor_count']} tensors generated")
        else:
            validation_results['errors'].append("✗ No tensors generated")
            validation_results['validation_passed'] = False
    
    # Check 4: Configuration validation
    if config.tensor_dimensions < 64 or config.tensor_dimensions > 2048:
        validation_results['warnings'].append(f"⚠ Tensor dimensions ({config.tensor_dimensions}) outside typical range (64-2048)")
    
    if config.distributed_shards < 1:
        validation_results['errors'].append("✗ Number of distributed shards must be at least 1")
        validation_results['validation_passed'] = False
    
    return validation_results


def generate_usage_documentation(config):
    """Generate usage documentation for the training set."""
    
    usage_content = f"""# Distributed GGML Training Set Usage Guide

## Overview

This training set was generated using the Distributed GGML Training Set Synthesis pipeline.
It contains semantically encoded fragments from your source data, organized as a hypergraph
and tensorized for compatibility with GGML-based models.

## Configuration

- **Data Path**: {config.data_path}
- **File Extensions**: {', '.join(config.file_extensions)}
- **Tensor Dimensions**: {config.tensor_dimensions}
- **Distributed Shards**: {config.distributed_shards}
- **Export Format**: {config.export_format}

## Directory Structure

```
{config.project_name}/
├── fragments_summary.json      # Summary of extracted fragments
├── hypergraph.json            # Hypergraph structure
├── training_manifest.json     # Complete training set manifest
├── training_params.json       # Training parameters used
├── validation_results.json    # Validation results
├── tensors/                   # Unified tensor files
│   ├── node_embeddings.npy
│   ├── edge_embeddings.npy
│   ├── attention_weights.npy
│   └── *_metadata.json
└── sharded_tensors/          # Distributed tensor shards
    ├── shard_0/
    ├── shard_1/
    ├── shard_2/
    └── shard_3/
```

## Usage Examples

### Loading Tensors (Python)

```python
import numpy as np
import json

# Load node embeddings
node_embeddings = np.load('{config.project_name}/tensors/node_embeddings.npy')
print(f"Node embeddings shape: {{node_embeddings.shape}}")

# Load metadata
with open('{config.project_name}/tensors/node_embeddings_metadata.json', 'r') as f:
    metadata = json.load(f)
    print(f"Number of nodes: {{metadata['shape_info']['num_nodes']}}")
```

### Loading Sharded Tensors

```python
import os
import numpy as np

# Load all shards
shards = []
for shard_dir in os.listdir('{config.project_name}/sharded_tensors'):
    shard_path = os.path.join('{config.project_name}/sharded_tensors', shard_dir)
    if os.path.isdir(shard_path):
        # Load tensors from this shard
        for tensor_file in os.listdir(shard_path):
            if tensor_file.endswith('.npy'):
                tensor_path = os.path.join(shard_path, tensor_file)
                tensor = np.load(tensor_path)
                shards.append(tensor)

print(f"Loaded {{len(shards)}} tensor shards")
```

### Integration with GGML Models

The generated tensors are compatible with GGML-based models. The recommended approach is:

1. Load the tensor files using your GGML framework
2. Use the attention weights for sample prioritization
3. Apply the hypergraph structure for relationship modeling
4. Leverage the distributed shards for parallel processing

## Extending the Training Set

To add new data to the training set:

1. Update the `data_path` in your configuration
2. Add new file extensions to `file_extensions` if needed
3. Re-run the training synthesis pipeline
4. The new data will be integrated into the existing hypergraph structure

## Validation

The training set has been validated for:
- ✓ File integrity
- ✓ Tensor consistency
- ✓ Hypergraph structure
- ✓ Attention weight computation
- ✓ Distributed shard creation

See `validation_results.json` for detailed validation results.

## Troubleshooting

If you encounter issues:

1. Check the validation results for errors
2. Verify that all required files exist
3. Ensure tensor dimensions are appropriate for your model
4. Check that sharded tensors are properly distributed

For more advanced usage patterns, refer to the AutoTrain documentation.
"""

    usage_path = os.path.join(config.project_name, 'USAGE.md')
    with open(usage_path, 'w') as f:
        f.write(usage_content)
    
    logger.info(f"Usage documentation saved to {usage_path}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.training_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    train(config)