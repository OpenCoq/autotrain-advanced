"""
Tensorization module for converting hypergraph to GGML-compatible tensors.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from autotrain import logger
from .hypergraph import HyperGraph, HyperNode, HyperEdge


class TensorEmbedding:
    """Represents a tensor embedding for a hypergraph element."""
    
    def __init__(self, element_id: str, embedding: np.ndarray, element_type: str, metadata: Dict[str, Any]):
        self.element_id = element_id
        self.embedding = embedding
        self.element_type = element_type  # 'node', 'edge'
        self.metadata = metadata
        
    def __repr__(self):
        return f"TensorEmbedding(id={self.element_id}, type={self.element_type}, shape={self.embedding.shape})"


class GGMLTensor:
    """Represents a GGML-compatible tensor."""
    
    def __init__(self, tensor_id: str, data: np.ndarray, tensor_type: str, metadata: Dict[str, Any]):
        self.tensor_id = tensor_id
        self.data = data
        self.tensor_type = tensor_type  # 'node_embeddings', 'edge_embeddings', 'attention_weights'
        self.metadata = metadata
        
    def __repr__(self):
        return f"GGMLTensor(id={self.tensor_id}, type={self.tensor_type}, shape={self.data.shape})"


class SemanticEncoder:
    """Encodes content into semantic embeddings."""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        
    def encode_content(self, content: str, content_type: str) -> np.ndarray:
        """Encode content into a semantic embedding vector."""
        # Simple hash-based encoding (in production, use transformer models)
        content_hash = hash(content)
        
        # Create a pseudo-random embedding based on content hash
        np.random.seed(abs(content_hash) % (2**32))
        embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Add type-specific bias
        type_bias = self._get_type_bias(content_type)
        embedding = embedding + type_bias
        
        return embedding.astype(np.float32)
        
    def _get_type_bias(self, content_type: str) -> np.ndarray:
        """Get type-specific bias for different content types."""
        type_biases = {
            'fragment': np.array([0.1] * self.embedding_dim),
            'concept': np.array([0.2] * self.embedding_dim),
            'code': np.array([0.3] * self.embedding_dim),
            'config': np.array([0.4] * self.embedding_dim),
            'doc': np.array([0.5] * self.embedding_dim),
            'text': np.array([0.6] * self.embedding_dim)
        }
        
        bias = type_biases.get(content_type, np.zeros(self.embedding_dim))
        return bias * 0.1  # Scale down the bias


class AttentionMechanism:
    """Implements ECAN-inspired attention mechanism."""
    
    def __init__(self, attention_threshold: float = 0.1):
        self.attention_threshold = attention_threshold
        
    def compute_attention_weights(self, embeddings: List[TensorEmbedding]) -> Dict[str, float]:
        """Compute attention weights for embeddings."""
        attention_weights = {}
        
        # Compute centrality-based attention
        for embedding in embeddings:
            attention_score = self._compute_attention_score(embedding, embeddings)
            
            # Apply threshold
            if attention_score > self.attention_threshold:
                attention_weights[embedding.element_id] = attention_score
                
        # Normalize weights
        if attention_weights:
            total_weight = sum(attention_weights.values())
            attention_weights = {k: v / total_weight for k, v in attention_weights.items()}
            
        logger.info(f"Computed attention weights for {len(attention_weights)} elements")
        return attention_weights
        
    def _compute_attention_score(self, target_embedding: TensorEmbedding, all_embeddings: List[TensorEmbedding]) -> float:
        """Compute attention score for a single embedding."""
        # Simple similarity-based attention
        similarities = []
        
        for other_embedding in all_embeddings:
            if other_embedding.element_id != target_embedding.element_id:
                similarity = self._compute_similarity(target_embedding.embedding, other_embedding.embedding)
                similarities.append(similarity)
                
        if similarities:
            # Attention score based on average similarity
            attention_score = np.mean(similarities)
            
            # Boost based on metadata
            metadata_boost = self._get_metadata_boost(target_embedding.metadata)
            attention_score *= metadata_boost
            
            return attention_score
            
        return 0.0
        
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
    def _get_metadata_boost(self, metadata: Dict[str, Any]) -> float:
        """Get attention boost based on metadata."""
        boost = 1.0
        
        # Boost based on element type
        element_type = metadata.get('element_type', '')
        if element_type == 'concept':
            boost *= 1.2
        elif element_type == 'fragment':
            boost *= 1.1
            
        # Boost based on frequency
        frequency = metadata.get('frequency', 1)
        boost *= min(1.0 + frequency * 0.1, 2.0)
        
        return boost


class TensorSharding:
    """Handles distributed sharding of tensors."""
    
    def __init__(self, num_shards: int = 4, use_semantic_clustering: bool = True):
        self.num_shards = num_shards
        self.use_semantic_clustering = use_semantic_clustering
        
    def shard_tensors(self, tensors: List[GGMLTensor]) -> Dict[int, List[GGMLTensor]]:
        """Shard tensors for distributed processing."""
        shards = defaultdict(list)
        
        if self.use_semantic_clustering:
            shard_assignments = self._semantic_clustering_sharding(tensors)
        else:
            shard_assignments = self._round_robin_sharding(tensors)
            
        for tensor, shard_id in zip(tensors, shard_assignments):
            shards[shard_id].append(tensor)
            
        logger.info(f"Sharded {len(tensors)} tensors into {len(shards)} shards")
        return dict(shards)
        
    def _semantic_clustering_sharding(self, tensors: List[GGMLTensor]) -> List[int]:
        """Shard tensors based on semantic clustering."""
        # Simple clustering based on tensor type
        type_to_shard = {}
        shard_assignments = []
        
        for tensor in tensors:
            tensor_type = tensor.tensor_type
            
            if tensor_type not in type_to_shard:
                type_to_shard[tensor_type] = len(type_to_shard) % self.num_shards
                
            shard_assignments.append(type_to_shard[tensor_type])
            
        return shard_assignments
        
    def _round_robin_sharding(self, tensors: List[GGMLTensor]) -> List[int]:
        """Simple round-robin sharding."""
        return [i % self.num_shards for i in range(len(tensors))]


class HyperGraphTensorizer:
    """Main tensorization class for converting hypergraph to GGML tensors."""
    
    def __init__(self, embedding_dim: int = 512, attention_threshold: float = 0.1, 
                 num_shards: int = 4, use_semantic_clustering: bool = True):
        self.encoder = SemanticEncoder(embedding_dim)
        self.attention = AttentionMechanism(attention_threshold)
        self.sharder = TensorSharding(num_shards, use_semantic_clustering)
        
    def tensorize_hypergraph(self, hypergraph: HyperGraph) -> Tuple[List[GGMLTensor], Dict[int, List[GGMLTensor]]]:
        """Convert hypergraph to GGML-compatible tensors."""
        logger.info(f"Starting tensorization of hypergraph with {len(hypergraph.nodes)} nodes and {len(hypergraph.edges)} edges")
        
        # Step 1: Create embeddings for nodes and edges
        embeddings = self._create_embeddings(hypergraph)
        
        # Step 2: Compute attention weights
        attention_weights = self.attention.compute_attention_weights(embeddings)
        
        # Step 3: Create GGML tensors
        tensors = self._create_ggml_tensors(embeddings, attention_weights)
        
        # Step 4: Shard tensors for distributed processing
        sharded_tensors = self.sharder.shard_tensors(tensors)
        
        logger.info(f"Created {len(tensors)} tensors and sharded into {len(sharded_tensors)} shards")
        return tensors, sharded_tensors
        
    def _create_embeddings(self, hypergraph: HyperGraph) -> List[TensorEmbedding]:
        """Create embeddings for all nodes and edges in the hypergraph."""
        embeddings = []
        
        # Create node embeddings
        for node in hypergraph.nodes.values():
            content = node.content
            content_type = node.node_type
            
            embedding = self.encoder.encode_content(content, content_type)
            
            metadata = {
                'element_type': 'node',
                'node_type': node.node_type,
                'attributes': node.attributes,
                'degree': len(node.edges)
            }
            
            tensor_embedding = TensorEmbedding(node.node_id, embedding, 'node', metadata)
            embeddings.append(tensor_embedding)
            
        # Create edge embeddings
        for edge in hypergraph.edges.values():
            # Edge content is a combination of connected node contents
            connected_nodes = [hypergraph.nodes[node_id] for node_id in edge.nodes if node_id in hypergraph.nodes]
            edge_content = ' '.join([node.content[:100] for node in connected_nodes])
            
            embedding = self.encoder.encode_content(edge_content, edge.edge_type)
            
            metadata = {
                'element_type': 'edge',
                'edge_type': edge.edge_type,
                'attributes': edge.attributes,
                'node_count': len(edge.nodes)
            }
            
            tensor_embedding = TensorEmbedding(edge.edge_id, embedding, 'edge', metadata)
            embeddings.append(tensor_embedding)
            
        return embeddings
        
    def _create_ggml_tensors(self, embeddings: List[TensorEmbedding], attention_weights: Dict[str, float]) -> List[GGMLTensor]:
        """Create GGML tensors from embeddings and attention weights."""
        tensors = []
        
        # Group embeddings by type
        node_embeddings = [emb for emb in embeddings if emb.element_type == 'node']
        edge_embeddings = [emb for emb in embeddings if emb.element_type == 'edge']
        
        # Create node embedding tensor
        if node_embeddings:
            node_data = np.stack([emb.embedding for emb in node_embeddings])
            node_metadata = {
                'element_ids': [emb.element_id for emb in node_embeddings],
                'element_types': [emb.metadata.get('node_type', 'unknown') for emb in node_embeddings],
                'shape_info': {
                    'num_nodes': len(node_embeddings),
                    'embedding_dim': node_embeddings[0].embedding.shape[0]
                }
            }
            
            node_tensor = GGMLTensor('node_embeddings', node_data, 'node_embeddings', node_metadata)
            tensors.append(node_tensor)
            
        # Create edge embedding tensor
        if edge_embeddings:
            edge_data = np.stack([emb.embedding for emb in edge_embeddings])
            edge_metadata = {
                'element_ids': [emb.element_id for emb in edge_embeddings],
                'element_types': [emb.metadata.get('edge_type', 'unknown') for emb in edge_embeddings],
                'shape_info': {
                    'num_edges': len(edge_embeddings),
                    'embedding_dim': edge_embeddings[0].embedding.shape[0]
                }
            }
            
            edge_tensor = GGMLTensor('edge_embeddings', edge_data, 'edge_embeddings', edge_metadata)
            tensors.append(edge_tensor)
            
        # Create attention weights tensor
        if attention_weights:
            attention_ids = list(attention_weights.keys())
            attention_data = np.array([attention_weights[id_] for id_ in attention_ids])
            attention_metadata = {
                'element_ids': attention_ids,
                'threshold': self.attention.attention_threshold,
                'shape_info': {
                    'num_elements': len(attention_ids)
                }
            }
            
            attention_tensor = GGMLTensor('attention_weights', attention_data, 'attention_weights', attention_metadata)
            tensors.append(attention_tensor)
            
        return tensors
        
    def export_tensors(self, tensors: List[GGMLTensor], output_dir: str):
        """Export tensors to GGML-compatible format."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export each tensor
        for tensor in tensors:
            tensor_path = os.path.join(output_dir, f"{tensor.tensor_id}.npy")
            np.save(tensor_path, tensor.data)
            
            # Export metadata
            metadata_path = os.path.join(output_dir, f"{tensor.tensor_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(tensor.metadata, f, indent=2)
                
        # Export summary
        summary = {
            'num_tensors': len(tensors),
            'tensor_types': list(set(tensor.tensor_type for tensor in tensors)),
            'total_size': sum(tensor.data.size for tensor in tensors),
            'shapes': {tensor.tensor_id: tensor.data.shape for tensor in tensors}
        }
        
        summary_path = os.path.join(output_dir, 'tensor_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Exported {len(tensors)} tensors to {output_dir}")
        
    def export_sharded_tensors(self, sharded_tensors: Dict[int, List[GGMLTensor]], output_dir: str):
        """Export sharded tensors to separate directories."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for shard_id, tensors in sharded_tensors.items():
            shard_dir = os.path.join(output_dir, f"shard_{shard_id}")
            self.export_tensors(tensors, shard_dir)
            
        logger.info(f"Exported {len(sharded_tensors)} shards to {output_dir}")