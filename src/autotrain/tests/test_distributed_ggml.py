"""
Tests for distributed GGML training set synthesis.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np

from autotrain.trainers.distributed_ggml.params import DistributedGGMLParams
from autotrain.trainers.distributed_ggml.file_parser import (
    FileDiscovery, FragmentParser, FileFragment, discover_and_parse_files
)
from autotrain.trainers.distributed_ggml.hypergraph import (
    HyperNode, HyperEdge, HyperGraph, HyperGraphEncoder
)
from autotrain.trainers.distributed_ggml.tensorizer import (
    SemanticEncoder, AttentionMechanism, TensorSharding, HyperGraphTensorizer
)
from autotrain.trainers.distributed_ggml.__main__ import train


class TestDistributedGGMLParams:
    """Test the parameters class."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = DistributedGGMLParams(data_path="/tmp/test")
        
        assert params.data_path == "/tmp/test"
        assert params.project_name == "distributed-ggml-training"
        assert params.tensor_dimensions == 512
        assert params.distributed_shards == 4
        assert params.semantic_clustering is True
        assert params.recursive_parsing is True
        
    def test_custom_params(self):
        """Test custom parameter values."""
        params = DistributedGGMLParams(
            data_path="/custom/path",
            project_name="custom-project",
            tensor_dimensions=256,
            distributed_shards=8,
            semantic_clustering=False
        )
        
        assert params.data_path == "/custom/path"
        assert params.project_name == "custom-project"
        assert params.tensor_dimensions == 256
        assert params.distributed_shards == 8
        assert params.semantic_clustering is False


class TestFileDiscovery:
    """Test file discovery functionality."""
    
    def test_file_discovery(self):
        """Test discovering files with specific extensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            (Path(tmp_dir) / "test.py").write_text("def hello(): pass")
            (Path(tmp_dir) / "config.yaml").write_text("key: value")
            (Path(tmp_dir) / "README.md").write_text("# Test")
            (Path(tmp_dir) / "ignored.txt").write_text("ignored")
            
            discovery = FileDiscovery(tmp_dir, [".py", ".yaml", ".md"])
            files = discovery.enumerate_files()
            
            assert len(files) == 3
            file_names = [f.name for f in files]
            assert "test.py" in file_names
            assert "config.yaml" in file_names
            assert "README.md" in file_names
            assert "ignored.txt" not in file_names


class TestFragmentParser:
    """Test fragment parsing functionality."""
    
    def test_parse_python_file(self):
        """Test parsing Python files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            python_file = Path(tmp_dir) / "test.py"
            python_content = """
def hello_world():
    '''Say hello to the world.'''
    print("Hello, World!")

class TestClass:
    def method(self):
        pass
"""
            python_file.write_text(python_content)
            
            parser = FragmentParser()
            fragments = parser.parse_file(python_file)
            
            assert len(fragments) >= 2  # At least function and class
            
            # Check that we have code fragments
            code_fragments = [f for f in fragments if f.fragment_type == 'code']
            assert len(code_fragments) >= 2
            
            # Check fragment content
            function_fragments = [f for f in code_fragments if f.metadata.get('name') == 'hello_world']
            assert len(function_fragments) == 1
            
            class_fragments = [f for f in code_fragments if f.metadata.get('name') == 'TestClass']
            assert len(class_fragments) == 1
            
    def test_parse_yaml_file(self):
        """Test parsing YAML files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_file = Path(tmp_dir) / "config.yaml"
            yaml_content = """
database:
  host: localhost
  port: 5432
  
app:
  name: test-app
  version: 1.0.0
"""
            yaml_file.write_text(yaml_content)
            
            parser = FragmentParser()
            fragments = parser.parse_file(yaml_file)
            
            assert len(fragments) >= 2  # At least database and app sections
            
            # Check that we have config fragments
            config_fragments = [f for f in fragments if f.fragment_type == 'config']
            assert len(config_fragments) >= 2
            
            # Check fragment keys
            keys = [f.metadata.get('key') for f in config_fragments]
            assert 'database' in keys
            assert 'app' in keys
            
    def test_parse_markdown_file(self):
        """Test parsing Markdown files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_file = Path(tmp_dir) / "README.md"
            md_content = """# Main Title

This is the introduction.

## Section 1

Content of section 1.

### Subsection 1.1

Content of subsection 1.1.

## Section 2

Content of section 2.
"""
            md_file.write_text(md_content)
            
            parser = FragmentParser()
            fragments = parser.parse_file(md_file)
            
            assert len(fragments) >= 3  # At least 3 sections
            
            # Check that we have doc fragments
            doc_fragments = [f for f in fragments if f.fragment_type == 'doc']
            assert len(doc_fragments) >= 3
            
            # Check fragment headers
            headers = [f.metadata.get('header') for f in doc_fragments]
            assert '# Main Title' in headers
            assert '## Section 1' in headers
            assert '## Section 2' in headers


class TestHyperGraph:
    """Test hypergraph functionality."""
    
    def test_hypergraph_creation(self):
        """Test creating hypergraph nodes and edges."""
        hypergraph = HyperGraph()
        
        # Create nodes
        node1 = HyperNode("node1", "content1", "fragment", {"type": "code"})
        node2 = HyperNode("node2", "content2", "concept", {"type": "function"})
        
        hypergraph.add_node(node1)
        hypergraph.add_node(node2)
        
        # Create edge
        edge = HyperEdge("edge1", "semantic", {"node1", "node2"}, {"weight": 1.0})
        hypergraph.add_edge(edge)
        
        assert len(hypergraph.nodes) == 2
        assert len(hypergraph.edges) == 1
        
        # Check node connections
        assert "edge1" in hypergraph.nodes["node1"].edges
        assert "edge1" in hypergraph.nodes["node2"].edges
        
    def test_hypergraph_queries(self):
        """Test querying hypergraph."""
        hypergraph = HyperGraph()
        
        # Add nodes
        node1 = HyperNode("node1", "content1", "fragment", {"type": "code"})
        node2 = HyperNode("node2", "content2", "concept", {"type": "function"})
        hypergraph.add_node(node1)
        hypergraph.add_node(node2)
        
        # Add edge
        edge = HyperEdge("edge1", "semantic", {"node1", "node2"}, {"weight": 1.0})
        hypergraph.add_edge(edge)
        
        # Test queries
        fragment_nodes = hypergraph.get_nodes_by_type("fragment")
        assert len(fragment_nodes) == 1
        assert fragment_nodes[0].node_id == "node1"
        
        concept_nodes = hypergraph.get_nodes_by_type("concept")
        assert len(concept_nodes) == 1
        assert concept_nodes[0].node_id == "node2"
        
        connected_nodes = hypergraph.get_connected_nodes("node1")
        assert len(connected_nodes) == 1
        assert connected_nodes[0].node_id == "node2"


class TestHyperGraphEncoder:
    """Test hypergraph encoding."""
    
    def test_encode_fragments(self):
        """Test encoding fragments into hypergraph."""
        # Create test fragments
        fragments = [
            FileFragment(
                "def hello(): pass",
                "code",
                {"file_path": "test.py", "type": "function"},
                {"name": "hello", "args": []}
            ),
            FileFragment(
                "class Test: pass",
                "code",
                {"file_path": "test.py", "type": "class"},
                {"name": "Test", "bases": []}
            )
        ]
        
        encoder = HyperGraphEncoder()
        hypergraph = encoder.encode_fragments(fragments)
        
        assert len(hypergraph.nodes) >= 2  # At least fragment nodes
        assert len(hypergraph.edges) >= 0  # May have edges
        
        # Check that fragment nodes were created
        fragment_nodes = hypergraph.get_nodes_by_type("fragment")
        assert len(fragment_nodes) == 2


class TestSemanticEncoder:
    """Test semantic encoding."""
    
    def test_encode_content(self):
        """Test encoding content into embeddings."""
        encoder = SemanticEncoder(embedding_dim=128)
        
        embedding = encoder.encode_content("test content", "code")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (128,)
        assert embedding.dtype == np.float32
        
        # Test that same content produces same embedding
        embedding2 = encoder.encode_content("test content", "code")
        np.testing.assert_array_equal(embedding, embedding2)
        
        # Test that different content produces different embeddings
        embedding3 = encoder.encode_content("different content", "code")
        assert not np.array_equal(embedding, embedding3)


class TestAttentionMechanism:
    """Test attention mechanism."""
    
    def test_compute_attention_weights(self):
        """Test computing attention weights."""
        from autotrain.trainers.distributed_ggml.tensorizer import TensorEmbedding
        
        attention = AttentionMechanism(attention_threshold=0.0)
        
        # Create test embeddings
        embeddings = [
            TensorEmbedding("emb1", np.random.rand(10), "node", {"element_type": "node"}),
            TensorEmbedding("emb2", np.random.rand(10), "node", {"element_type": "node"}),
            TensorEmbedding("emb3", np.random.rand(10), "edge", {"element_type": "edge"})
        ]
        
        weights = attention.compute_attention_weights(embeddings)
        
        assert isinstance(weights, dict)
        assert len(weights) <= len(embeddings)
        
        # Check that weights sum to 1 (if any weights exist)
        if weights:
            total_weight = sum(weights.values())
            assert abs(total_weight - 1.0) < 1e-6


class TestTensorSharding:
    """Test tensor sharding."""
    
    def test_shard_tensors(self):
        """Test sharding tensors."""
        from autotrain.trainers.distributed_ggml.tensorizer import GGMLTensor
        
        sharder = TensorSharding(num_shards=2)
        
        # Create test tensors
        tensors = [
            GGMLTensor("tensor1", np.random.rand(10, 5), "node_embeddings", {}),
            GGMLTensor("tensor2", np.random.rand(8, 5), "edge_embeddings", {}),
            GGMLTensor("tensor3", np.random.rand(12, 5), "node_embeddings", {}),
            GGMLTensor("tensor4", np.random.rand(6, 5), "attention_weights", {})
        ]
        
        sharded_tensors = sharder.shard_tensors(tensors)
        
        assert len(sharded_tensors) <= 2  # At most 2 shards
        
        # Check that all tensors are assigned to shards
        total_tensors = sum(len(shard) for shard in sharded_tensors.values())
        assert total_tensors == len(tensors)


class TestHyperGraphTensorizer:
    """Test hypergraph tensorization."""
    
    def test_tensorize_hypergraph(self):
        """Test tensorizing hypergraph."""
        # Create test hypergraph
        hypergraph = HyperGraph()
        
        node1 = HyperNode("node1", "content1", "fragment", {"type": "code"})
        node2 = HyperNode("node2", "content2", "concept", {"type": "function"})
        hypergraph.add_node(node1)
        hypergraph.add_node(node2)
        
        edge = HyperEdge("edge1", "semantic", {"node1", "node2"}, {"weight": 1.0})
        hypergraph.add_edge(edge)
        
        tensorizer = HyperGraphTensorizer(
            embedding_dim=64,
            attention_threshold=0.0,
            num_shards=2
        )
        
        tensors, sharded_tensors = tensorizer.tensorize_hypergraph(hypergraph)
        
        assert len(tensors) > 0
        assert len(sharded_tensors) > 0
        
        # Check tensor types
        tensor_types = [tensor.tensor_type for tensor in tensors]
        assert "node_embeddings" in tensor_types or "edge_embeddings" in tensor_types


class TestTrainingPipeline:
    """Test the complete training pipeline."""
    
    def test_training_pipeline(self):
        """Test the complete training pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test data
            test_data_dir = Path(tmp_dir) / "test_data"
            test_data_dir.mkdir()
            
            (test_data_dir / "test.py").write_text("""
def hello():
    '''Hello function.'''
    print("Hello, World!")

class TestClass:
    def method(self):
        return "test"
""")
            
            (test_data_dir / "config.yaml").write_text("""
app:
  name: test-app
  version: 1.0.0
""")
            
            # Create output directory
            output_dir = Path(tmp_dir) / "output"
            
            # Create configuration
            config = DistributedGGMLParams(
                data_path=str(test_data_dir),
                project_name=str(output_dir),
                tensor_dimensions=64,
                distributed_shards=2,
                file_extensions=[".py", ".yaml"]
            )
            
            # Run training
            train(config)
            
            # Check outputs
            assert output_dir.exists()
            assert (output_dir / "fragments_summary.json").exists()
            assert (output_dir / "hypergraph.json").exists()
            assert (output_dir / "training_manifest.json").exists()
            assert (output_dir / "validation_results.json").exists()
            assert (output_dir / "USAGE.md").exists()
            assert (output_dir / "tensors").exists()
            assert (output_dir / "sharded_tensors").exists()
            
            # Check validation results
            with open(output_dir / "validation_results.json") as f:
                validation = json.load(f)
                assert validation["validation_passed"] is True
                
            # Check manifest
            with open(output_dir / "training_manifest.json") as f:
                manifest = json.load(f)
                assert manifest["statistics"]["total_fragments"] > 0
                assert manifest["statistics"]["tensor_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__])