from typing import Optional

from pydantic import Field, BaseModel


class AutoTrainParams(BaseModel):
    """Base class for AutoTrain parameters."""
    
    class Config:
        protected_namespaces = ()
        
    def model_dump(self, **kwargs):
        """Get model dict."""
        # Use the superclass method from pydantic v2
        return super().model_dump(**kwargs)
        
    def model_dump_json(self, **kwargs):
        """Get model JSON."""
        return super().model_dump_json(**kwargs)


class DistributedGGMLParams(AutoTrainParams):
    """
    [`DistributedGGMLParams`] is a configuration class for distributed GGML training set synthesis parameters.

    Attributes:
        data_path (str): Path to the dataset/repository to process.
        project_name (str): Name of the project. Default is "distributed-ggml-training".
        file_extensions (list): List of file extensions to process. Default includes common code and doc files.
        max_fragment_size (int): Maximum size of semantic fragments. Default is 1024.
        tensor_dimensions (int): Number of dimensions for tensor embeddings. Default is 512.
        attention_threshold (float): Threshold for attention weight filtering. Default is 0.1.
        distributed_shards (int): Number of distributed shards to create. Default is 4.
        hypergraph_depth (int): Maximum depth for hypergraph relationships. Default is 3.
        export_format (str): Export format for the training set. Default is "ggml".
        semantic_clustering (bool): Whether to use semantic clustering for sharding. Default is True.
        recursive_parsing (bool): Whether to use recursive parsing for nested structures. Default is True.
    """

    data_path: str = Field(None, title="Data path")
    project_name: str = Field("distributed-ggml-training", title="Output directory")
    file_extensions: list = Field(
        default_factory=lambda: [".py", ".md", ".yaml", ".yml", ".json", ".txt", ".rst", ".toml", ".cfg", ".ini"],
        title="File extensions to process"
    )
    max_fragment_size: int = Field(1024, title="Maximum fragment size")
    tensor_dimensions: int = Field(512, title="Tensor embedding dimensions")
    attention_threshold: float = Field(0.1, title="Attention threshold")
    distributed_shards: int = Field(4, title="Number of distributed shards")
    hypergraph_depth: int = Field(3, title="Hypergraph relationship depth")
    export_format: str = Field("ggml", title="Export format")
    semantic_clustering: bool = Field(True, title="Use semantic clustering")
    recursive_parsing: bool = Field(True, title="Use recursive parsing")
    
    # Base parameters from AutoTrainParams
    token: Optional[str] = Field(None, title="Hub Token")
    push_to_hub: bool = Field(False, title="Push to hub")
    username: Optional[str] = Field(None, title="Hugging Face Username")
    log: str = Field("none", title="Logging using experiment tracking")