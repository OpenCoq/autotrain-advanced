#!/usr/bin/env python
"""
Simple test script for distributed GGML training set synthesis.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

from autotrain.trainers.distributed_ggml.params import DistributedGGMLParams
from autotrain.trainers.distributed_ggml.__main__ import train


def test_basic_functionality():
    """Test basic functionality of the distributed GGML trainer."""
    
    print("Testing Distributed GGML Training Set Synthesis")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test data directory
        test_data_dir = Path(tmp_dir) / "test_data"
        test_data_dir.mkdir()
        
        # Create sample files
        (test_data_dir / "sample.py").write_text("""
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "Hello, World!"

class Calculator:
    '''A simple calculator class.'''
    
    def add(self, a, b):
        return a + b
        
    def subtract(self, a, b):
        return a - b
""")
        
        (test_data_dir / "config.yaml").write_text("""
app:
  name: test-application
  version: 1.0.0
  
database:
  host: localhost
  port: 5432
  name: testdb
""")
        
        (test_data_dir / "README.md").write_text("""
# Test Project

This is a test project for demonstrating the distributed GGML training set synthesis.

## Features

- File parsing
- Hypergraph construction
- Tensorization
- Distributed sharding

## Usage

Run the training synthesis to generate the training set.
""")
        
        # Create output directory
        output_dir = Path(tmp_dir) / "output"
        
        # Create configuration
        config = DistributedGGMLParams(
            data_path=str(test_data_dir),
            project_name=str(output_dir),
            tensor_dimensions=64,  # Smaller for testing
            distributed_shards=2,   # Fewer shards for testing
            file_extensions=[".py", ".yaml", ".md"]
        )
        
        print(f"Input data path: {config.data_path}")
        print(f"Output directory: {config.project_name}")
        print(f"Processing files: {config.file_extensions}")
        print()
        
        # Run training
        try:
            train(config)
            print("\n✅ Training completed successfully!")
            
            # Check outputs
            expected_files = [
                "fragments_summary.json",
                "hypergraph.json", 
                "training_manifest.json",
                "validation_results.json",
                "training_params.json",
                "USAGE.md"
            ]
            
            print("\nChecking output files:")
            for file_name in expected_files:
                file_path = output_dir / file_name
                if file_path.exists():
                    print(f"✅ {file_name} - exists")
                else:
                    print(f"❌ {file_name} - missing")
                    
            # Check directories
            if (output_dir / "tensors").exists():
                print("✅ tensors/ directory - exists")
            else:
                print("❌ tensors/ directory - missing")
                
            if (output_dir / "sharded_tensors").exists():
                print("✅ sharded_tensors/ directory - exists")
            else:
                print("❌ sharded_tensors/ directory - missing")
            
            # Show summary
            print("\n📊 Training Summary:")
            manifest_path = output_dir / "training_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    stats = manifest.get("statistics", {})
                    print(f"  Total fragments: {stats.get('total_fragments', 0)}")
                    print(f"  Hypergraph nodes: {stats.get('hypergraph_nodes', 0)}")
                    print(f"  Hypergraph edges: {stats.get('hypergraph_edges', 0)}")
                    print(f"  Generated tensors: {stats.get('tensor_count', 0)}")
                    print(f"  Distributed shards: {stats.get('shard_count', 0)}")
            
            # Show validation results
            print("\n🔍 Validation Results:")
            validation_path = output_dir / "validation_results.json"
            if validation_path.exists():
                with open(validation_path) as f:
                    validation = json.load(f)
                    if validation.get("validation_passed", False):
                        print("✅ All validations passed")
                    else:
                        print("❌ Some validations failed")
                        
                    for check in validation.get("checks", []):
                        print(f"  {check}")
                        
                    for warning in validation.get("warnings", []):
                        print(f"  {warning}")
                        
                    for error in validation.get("errors", []):
                        print(f"  {error}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)