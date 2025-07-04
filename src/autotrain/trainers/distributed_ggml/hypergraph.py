"""
Hypergraph encoding module for distributed GGML training set synthesis.
"""

import json
import hashlib
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict

from autotrain import logger
from .file_parser import FileFragment


class HyperNode:
    """Represents a node in the hypergraph."""
    
    def __init__(self, node_id: str, content: str, node_type: str, attributes: Dict[str, Any]):
        self.node_id = node_id
        self.content = content
        self.node_type = node_type  # 'fragment', 'concept', 'relationship'
        self.attributes = attributes
        self.edges = set()  # Connected edge IDs
        
    def add_edge(self, edge_id: str):
        """Add an edge connection to this node."""
        self.edges.add(edge_id)
        
    def __repr__(self):
        return f"HyperNode(id={self.node_id}, type={self.node_type}, edges={len(self.edges)})"


class HyperEdge:
    """Represents an edge in the hypergraph connecting multiple nodes."""
    
    def __init__(self, edge_id: str, edge_type: str, nodes: Set[str], attributes: Dict[str, Any]):
        self.edge_id = edge_id
        self.edge_type = edge_type  # 'dependency', 'semantic', 'structural'
        self.nodes = nodes  # Set of node IDs
        self.attributes = attributes
        
    def __repr__(self):
        return f"HyperEdge(id={self.edge_id}, type={self.edge_type}, nodes={len(self.nodes)})"


class HyperGraph:
    """Represents a hypergraph structure for semantic fragments."""
    
    def __init__(self):
        self.nodes: Dict[str, HyperNode] = {}
        self.edges: Dict[str, HyperEdge] = {}
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # Type -> Node IDs
        
    def add_node(self, node: HyperNode):
        """Add a node to the hypergraph."""
        self.nodes[node.node_id] = node
        self.node_index[node.node_type].add(node.node_id)
        
    def add_edge(self, edge: HyperEdge):
        """Add an edge to the hypergraph."""
        self.edges[edge.edge_id] = edge
        
        # Update node connections
        for node_id in edge.nodes:
            if node_id in self.nodes:
                self.nodes[node_id].add_edge(edge.edge_id)
                
    def get_node(self, node_id: str) -> HyperNode:
        """Get a node by ID."""
        return self.nodes.get(node_id)
        
    def get_edge(self, edge_id: str) -> HyperEdge:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
        
    def get_nodes_by_type(self, node_type: str) -> List[HyperNode]:
        """Get all nodes of a specific type."""
        return [self.nodes[node_id] for node_id in self.node_index[node_type]]
        
    def get_connected_nodes(self, node_id: str) -> List[HyperNode]:
        """Get all nodes connected to a given node."""
        connected = []
        node = self.nodes.get(node_id)
        if not node:
            return connected
            
        for edge_id in node.edges:
            edge = self.edges.get(edge_id)
            if edge:
                for connected_node_id in edge.nodes:
                    if connected_node_id != node_id and connected_node_id in self.nodes:
                        connected.append(self.nodes[connected_node_id])
                        
        return connected
        
    def __repr__(self):
        return f"HyperGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


class HyperGraphEncoder:
    """Encodes file fragments into a hypergraph structure."""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.hypergraph = HyperGraph()
        
    def encode_fragments(self, fragments: List[FileFragment]) -> HyperGraph:
        """Encode a list of file fragments into a hypergraph."""
        logger.info(f"Encoding {len(fragments)} fragments into hypergraph")
        
        # Step 1: Convert fragments to nodes
        self._create_fragment_nodes(fragments)
        
        # Step 2: Extract concepts and create concept nodes
        self._create_concept_nodes(fragments)
        
        # Step 3: Create relationships between nodes
        self._create_relationships()
        
        logger.info(f"Created hypergraph with {len(self.hypergraph.nodes)} nodes and {len(self.hypergraph.edges)} edges")
        return self.hypergraph
        
    def _create_fragment_nodes(self, fragments: List[FileFragment]):
        """Create nodes for each fragment."""
        for fragment in fragments:
            node_id = self._generate_node_id(fragment.content, fragment.fragment_type)
            
            attributes = {
                'fragment_type': fragment.fragment_type,
                'context': fragment.context,
                'metadata': fragment.metadata,
                'content_hash': hashlib.md5(fragment.content.encode()).hexdigest(),
                'content_length': len(fragment.content)
            }
            
            node = HyperNode(node_id, fragment.content, 'fragment', attributes)
            self.hypergraph.add_node(node)
            
    def _create_concept_nodes(self, fragments: List[FileFragment]):
        """Extract concepts from fragments and create concept nodes."""
        concepts = defaultdict(list)
        
        for fragment in fragments:
            # Extract key concepts based on fragment type
            if fragment.fragment_type == 'code':
                concepts['function'].extend(self._extract_function_concepts(fragment))
                concepts['class'].extend(self._extract_class_concepts(fragment))
                concepts['import'].extend(self._extract_import_concepts(fragment))
            elif fragment.fragment_type == 'config':
                concepts['configuration'].extend(self._extract_config_concepts(fragment))
            elif fragment.fragment_type == 'doc':
                concepts['documentation'].extend(self._extract_doc_concepts(fragment))
                
        # Create concept nodes
        for concept_type, concept_list in concepts.items():
            for concept in set(concept_list):  # Remove duplicates
                node_id = self._generate_node_id(concept, concept_type)
                
                attributes = {
                    'concept_type': concept_type,
                    'concept_name': concept,
                    'frequency': concept_list.count(concept)
                }
                
                node = HyperNode(node_id, concept, 'concept', attributes)
                self.hypergraph.add_node(node)
                
    def _create_relationships(self):
        """Create relationships between nodes."""
        fragment_nodes = self.hypergraph.get_nodes_by_type('fragment')
        concept_nodes = self.hypergraph.get_nodes_by_type('concept')
        
        # Create fragment-concept relationships
        for fragment_node in fragment_nodes:
            for concept_node in concept_nodes:
                if self._are_related(fragment_node, concept_node):
                    self._create_edge(fragment_node, concept_node, 'semantic')
                    
        # Create fragment-fragment relationships
        for i, fragment1 in enumerate(fragment_nodes):
            for fragment2 in fragment_nodes[i+1:]:
                if self._are_fragments_related(fragment1, fragment2):
                    self._create_edge(fragment1, fragment2, 'structural')
                    
    def _are_related(self, fragment_node: HyperNode, concept_node: HyperNode) -> bool:
        """Check if a fragment and concept are related."""
        concept_name = concept_node.attributes.get('concept_name', '')
        fragment_content = fragment_node.content.lower()
        
        # Simple keyword matching (can be improved with more sophisticated NLP)
        return concept_name.lower() in fragment_content
        
    def _are_fragments_related(self, fragment1: HyperNode, fragment2: HyperNode) -> bool:
        """Check if two fragments are related."""
        # Check if fragments are from the same file
        file1 = fragment1.attributes.get('context', {}).get('file_path', '')
        file2 = fragment2.attributes.get('context', {}).get('file_path', '')
        
        if file1 == file2:
            return True
            
        # Check for common imports or dependencies
        if fragment1.attributes.get('fragment_type') == 'code' and fragment2.attributes.get('fragment_type') == 'code':
            meta1 = fragment1.attributes.get('metadata', {})
            meta2 = fragment2.attributes.get('metadata', {})
            
            # Check for common function/class names
            name1 = meta1.get('name', '')
            name2 = meta2.get('name', '')
            
            if name1 and name2 and (name1 in fragment2.content or name2 in fragment1.content):
                return True
                
        return False
        
    def _create_edge(self, node1: HyperNode, node2: HyperNode, edge_type: str):
        """Create an edge between two nodes."""
        edge_id = f"{node1.node_id}_{node2.node_id}_{edge_type}"
        nodes = {node1.node_id, node2.node_id}
        
        attributes = {
            'edge_type': edge_type,
            'weight': 1.0  # Can be calculated based on relationship strength
        }
        
        edge = HyperEdge(edge_id, edge_type, nodes, attributes)
        self.hypergraph.add_edge(edge)
        
    def _extract_function_concepts(self, fragment: FileFragment) -> List[str]:
        """Extract function-related concepts from a code fragment."""
        concepts = []
        metadata = fragment.metadata
        
        if metadata.get('name'):
            concepts.append(metadata['name'])
            
        if metadata.get('args'):
            concepts.extend(metadata['args'])
            
        return concepts
        
    def _extract_class_concepts(self, fragment: FileFragment) -> List[str]:
        """Extract class-related concepts from a code fragment."""
        concepts = []
        metadata = fragment.metadata
        
        if metadata.get('name'):
            concepts.append(metadata['name'])
            
        if metadata.get('bases'):
            concepts.extend(metadata['bases'])
            
        return concepts
        
    def _extract_import_concepts(self, fragment: FileFragment) -> List[str]:
        """Extract import-related concepts from a code fragment."""
        concepts = []
        content = fragment.content
        
        # Simple regex-based extraction (can be improved)
        import_lines = [line for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
        
        for line in import_lines:
            if 'import' in line:
                parts = line.split()
                if len(parts) >= 2:
                    concepts.append(parts[1].split('.')[0])
                    
        return concepts
        
    def _extract_config_concepts(self, fragment: FileFragment) -> List[str]:
        """Extract configuration-related concepts from a config fragment."""
        concepts = []
        metadata = fragment.metadata
        
        if metadata.get('key'):
            concepts.append(metadata['key'])
            
        return concepts
        
    def _extract_doc_concepts(self, fragment: FileFragment) -> List[str]:
        """Extract documentation-related concepts from a doc fragment."""
        concepts = []
        metadata = fragment.metadata
        
        if metadata.get('header'):
            # Extract words from header
            header = metadata['header'].replace('#', '').strip()
            words = [word.strip() for word in header.split() if len(word.strip()) > 2]
            concepts.extend(words)
            
        return concepts
        
    def _generate_node_id(self, content: str, node_type: str) -> str:
        """Generate a unique node ID based on content and type."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{node_type}_{content_hash}"
        
    def export_hypergraph(self, output_path: str):
        """Export hypergraph to JSON format."""
        export_data = {
            'nodes': [],
            'edges': []
        }
        
        for node in self.hypergraph.nodes.values():
            export_data['nodes'].append({
                'id': node.node_id,
                'type': node.node_type,
                'content': node.content[:100] + '...' if len(node.content) > 100 else node.content,
                'attributes': node.attributes
            })
            
        for edge in self.hypergraph.edges.values():
            export_data['edges'].append({
                'id': edge.edge_id,
                'type': edge.edge_type,
                'nodes': list(edge.nodes),
                'attributes': edge.attributes
            })
            
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported hypergraph to {output_path}")