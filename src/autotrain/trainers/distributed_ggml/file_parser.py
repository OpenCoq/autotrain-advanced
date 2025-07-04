"""
File discovery and parsing module for distributed GGML training set synthesis.
"""

import os
import ast
import json
import yaml
from typing import List, Dict, Any, Tuple
from pathlib import Path

from autotrain import logger


class FileFragment:
    """Represents a semantic fragment extracted from a file."""
    
    def __init__(self, content: str, fragment_type: str, context: Dict[str, Any], metadata: Dict[str, Any]):
        self.content = content
        self.fragment_type = fragment_type  # 'code', 'text', 'config', 'doc'
        self.context = context  # file origin, line numbers, etc.
        self.metadata = metadata  # type signatures, dependencies, etc.
        
    def __repr__(self):
        return f"FileFragment(type={self.fragment_type}, len={len(self.content)}, file={self.context.get('file_path', 'unknown')})"


class FileDiscovery:
    """Discovers and enumerates files for processing."""
    
    def __init__(self, root_path: str, file_extensions: List[str]):
        self.root_path = Path(root_path)
        self.file_extensions = file_extensions
        
    def enumerate_files(self) -> List[Path]:
        """Recursively enumerate all files with relevant extensions."""
        files = []
        for ext in self.file_extensions:
            pattern = f"**/*{ext}"
            files.extend(self.root_path.glob(pattern))
        
        logger.info(f"Found {len(files)} files to process")
        return files


class FragmentParser:
    """Parses files and extracts semantic fragments."""
    
    def __init__(self, max_fragment_size: int = 1024):
        self.max_fragment_size = max_fragment_size
        
    def parse_file(self, file_path: Path) -> List[FileFragment]:
        """Parse a single file and extract semantic fragments."""
        fragments = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return fragments
            
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.py':
            fragments.extend(self._parse_python_file(file_path, content))
        elif file_ext in ['.yaml', '.yml']:
            fragments.extend(self._parse_yaml_file(file_path, content))
        elif file_ext == '.json':
            fragments.extend(self._parse_json_file(file_path, content))
        elif file_ext == '.md':
            fragments.extend(self._parse_markdown_file(file_path, content))
        else:
            fragments.extend(self._parse_text_file(file_path, content))
            
        return fragments
        
    def _parse_python_file(self, file_path: Path, content: str) -> List[FileFragment]:
        """Parse Python files and extract functions, classes, and docstrings."""
        fragments = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_source = ast.get_source_segment(content, node)
                    if func_source:
                        context = {
                            'file_path': str(file_path),
                            'line_start': node.lineno,
                            'line_end': node.end_lineno,
                            'type': 'function'
                        }
                        metadata = {
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'returns': ast.unparse(node.returns) if node.returns else None,
                            'decorators': [ast.unparse(dec) for dec in node.decorator_list]
                        }
                        fragments.append(FileFragment(func_source, 'code', context, metadata))
                        
                elif isinstance(node, ast.ClassDef):
                    class_source = ast.get_source_segment(content, node)
                    if class_source:
                        context = {
                            'file_path': str(file_path),
                            'line_start': node.lineno,
                            'line_end': node.end_lineno,
                            'type': 'class'
                        }
                        metadata = {
                            'name': node.name,
                            'bases': [ast.unparse(base) for base in node.bases],
                            'decorators': [ast.unparse(dec) for dec in node.decorator_list]
                        }
                        fragments.append(FileFragment(class_source, 'code', context, metadata))
                        
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            # Fall back to text parsing
            fragments.extend(self._parse_text_file(file_path, content))
            
        return fragments
        
    def _parse_yaml_file(self, file_path: Path, content: str) -> List[FileFragment]:
        """Parse YAML files and extract configuration sections."""
        fragments = []
        
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                for key, value in data.items():
                    section_content = yaml.dump({key: value}, default_flow_style=False)
                    context = {
                        'file_path': str(file_path),
                        'type': 'config_section'
                    }
                    metadata = {
                        'key': key,
                        'value_type': type(value).__name__
                    }
                    fragments.append(FileFragment(section_content, 'config', context, metadata))
        except Exception as e:
            logger.warning(f"Failed to parse YAML file {file_path}: {e}")
            fragments.extend(self._parse_text_file(file_path, content))
            
        return fragments
        
    def _parse_json_file(self, file_path: Path, content: str) -> List[FileFragment]:
        """Parse JSON files and extract key-value pairs."""
        fragments = []
        
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                for key, value in data.items():
                    section_content = json.dumps({key: value}, indent=2)
                    context = {
                        'file_path': str(file_path),
                        'type': 'config_section'
                    }
                    metadata = {
                        'key': key,
                        'value_type': type(value).__name__
                    }
                    fragments.append(FileFragment(section_content, 'config', context, metadata))
        except Exception as e:
            logger.warning(f"Failed to parse JSON file {file_path}: {e}")
            fragments.extend(self._parse_text_file(file_path, content))
            
        return fragments
        
    def _parse_markdown_file(self, file_path: Path, content: str) -> List[FileFragment]:
        """Parse Markdown files and extract sections."""
        fragments = []
        
        lines = content.split('\n')
        current_section = []
        section_header = None
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Save previous section
                if current_section and section_header:
                    section_content = '\n'.join(current_section)
                    context = {
                        'file_path': str(file_path),
                        'type': 'markdown_section'
                    }
                    metadata = {
                        'header': section_header,
                        'level': section_header.count('#')
                    }
                    fragments.append(FileFragment(section_content, 'doc', context, metadata))
                
                # Start new section
                section_header = line
                current_section = [line]
            else:
                current_section.append(line)
                
        # Save last section
        if current_section and section_header:
            section_content = '\n'.join(current_section)
            context = {
                'file_path': str(file_path),
                'type': 'markdown_section'
            }
            metadata = {
                'header': section_header,
                'level': section_header.count('#')
            }
            fragments.append(FileFragment(section_content, 'doc', context, metadata))
            
        return fragments
        
    def _parse_text_file(self, file_path: Path, content: str) -> List[FileFragment]:
        """Parse generic text files by splitting into chunks."""
        fragments = []
        
        # Split content into chunks of max_fragment_size
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > self.max_fragment_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                context = {
                    'file_path': str(file_path),
                    'type': 'text_chunk'
                }
                metadata = {
                    'chunk_size': len(chunk_content),
                    'line_count': len(current_chunk)
                }
                fragments.append(FileFragment(chunk_content, 'text', context, metadata))
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)
                
        # Add last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            context = {
                'file_path': str(file_path),
                'type': 'text_chunk'
            }
            metadata = {
                'chunk_size': len(chunk_content),
                'line_count': len(current_chunk)
            }
            fragments.append(FileFragment(chunk_content, 'text', context, metadata))
            
        return fragments


def discover_and_parse_files(root_path: str, file_extensions: List[str], max_fragment_size: int = 1024) -> List[FileFragment]:
    """Main function to discover and parse files into fragments."""
    logger.info(f"Starting file discovery and parsing in {root_path}")
    
    # Discover files
    discovery = FileDiscovery(root_path, file_extensions)
    files = discovery.enumerate_files()
    
    # Parse files
    parser = FragmentParser(max_fragment_size)
    all_fragments = []
    
    for file_path in files:
        fragments = parser.parse_file(file_path)
        all_fragments.extend(fragments)
        
    logger.info(f"Extracted {len(all_fragments)} semantic fragments from {len(files)} files")
    return all_fragments