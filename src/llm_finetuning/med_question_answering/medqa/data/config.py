from pydantic import BaseModel


class MedQACorpusChunkingConfig(BaseModel):
    """Configuration for MedQA corpus chunking."""

    corpus_dir: str = "textbooks/en"
    chunked_dataset_file: str = "medqa_textbooks_chunked.jsonl"
    hf_dataset_dir: str = "medqa_textbooks_chunked"
    hf_dataset_name: str = "awinml/medqa-textbooks-chunked"
    hf_private: bool = False
    partition_by: str = "title"
    file_extension: str = ".txt"


class MedQAIndexingConfig(BaseModel):
    """Configuration for MedQA corpus indexing."""

    # Milvus connection
    milvus_uri: str
    milvus_token: str
    collection_name: str = "medqa_textbooks"

    # Schema fields
    id_field: str = "id"
    file_name_field: str = "file_name"
    content_field: str = "content"
    sparse_vector_field: str = "sparse_vector"
    dense_vector_field: str = "dense_vector"
    metadata_field: str = "metadata"

    # Field parameters
    id_max_length: int = 1000
    file_name_max_length: int = 2550
    content_max_length: int = 65535
    dense_vector_dim: int = 1024

    # Analyzer parameters
    tokenizer: str = "standard"
    filters: list[str] = ["lowercase"]

    # Index parameters
    sparse_index_type: str = "SPARSE_INVERTED_INDEX"
    sparse_metric_type: str = "BM25"
    dense_index_type: str = "FLAT"
    dense_metric_type: str = "IP"

    # BM25 function
    bm25_function_name: str = "bm25"
    bm25_function_type: str = "BM25"

    # Model parameters
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    padding_side: str = "left"

    # Processing parameters
    batch_size: int = 256
    target_devices: list[str] = ["cuda:0", "cuda:1"]

    # Dataset parameters
    hf_dataset_name: str = "awinml/medqa-textbooks-chunked"
    dataset_split: str = "train"


class MedQADatasetProcessingConfig(BaseModel):
    """Configuration for MedQA dataset processing."""

    # File paths
    train_file: str = "questions/US/4_options/phrases_no_exclude_train.jsonl"
    dev_file: str = "questions/US/4_options/phrases_no_exclude_dev.jsonl"
    test_file: str = "questions/US/4_options/phrases_no_exclude_test.jsonl"

    # Dataset parameters
    hf_dataset_name: str = "awinml/medqa"
    hf_private: bool = False

    # Split names
    train_split_name: str = "train"
    validation_split_name: str = "validation"
    test_split_name: str = "test"


class MedQAContextAdditionConfig(BaseModel):
    """Configuration for MedQA context addition."""

    # Milvus connection
    milvus_uri: str
    milvus_token: str
    collection_name: str = "medqa_textbooks"

    # Model parameters
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    padding_side: str = "left"

    # Dataset parameters
    input_hf_dataset_name: str = "awinml/medqa"
    output_hf_dataset_name: str = "awinml/medqa-context"
    hf_private: bool = False

    # Field names
    question_field: str = "question"
    context_bm25_field: str = "context_bm25"
    context_qwen_embed_field: str = "context_qwen_embed"

    # Retrieval parameters
    top_k: int = 10
    batch_size: int = 32

    # Split names
    train_split_name: str = "train"
    validation_split_name: str = "validation"
    test_split_name: str = "test"
