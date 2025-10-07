import logging
from typing import Any

import tqdm
from config import MedQAIndexingConfig
from datasets import load_dataset
from pymilvus import (
    DataType,
    Function,
    FunctionType,
    MilvusClient,
)
from sentence_transformers import SentenceTransformer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedQAIndexer:
    """A class to index MedQA corpus into Milvus database."""

    def __init__(self, config: MedQAIndexingConfig):
        """Initialize the MedQAIndexer with configuration.

        Args:
            config (MedQAIndexingConfig): Configuration object
        """
        self.config = config
        self.client = None
        self.model = None
        self.pool = None

    def connect_to_milvus(self) -> None:
        """Connect to Milvus database."""
        self.client = MilvusClient(
            uri=self.config.milvus_uri, token=self.config.milvus_token
        )
        logger.info("Connected to Milvus successfully")

    def define_schema(self) -> Any:
        """Define the schema for the Milvus collection.

        Returns:
            Any: Milvus schema object
        """
        # Define tokenizer parameters for text analysis
        analyzer_params = {
            "tokenizer": self.config.tokenizer,
            "filter": self.config.filters,
        }

        # Create schema
        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name=self.config.id_field,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=self.config.id_max_length,
        )
        schema.add_field(
            field_name=self.config.file_name_field,
            datatype=DataType.VARCHAR,
            max_length=self.config.file_name_max_length,
        )
        schema.add_field(
            field_name=self.config.content_field,
            datatype=DataType.VARCHAR,
            max_length=self.config.content_max_length,
            analyzer_params=analyzer_params,
            enable_match=True,  # Enable text matching
            enable_analyzer=True,  # Enable text analysis
        )
        schema.add_field(
            field_name=self.config.sparse_vector_field,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name=self.config.dense_vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self.config.dense_vector_dim,
        )
        schema.add_field(field_name=self.config.metadata_field, datatype=DataType.JSON)

        return schema

    def define_bm25_function(self) -> Function:
        """Define BM25 function to generate sparse vectors from text.

        Returns:
            Function: BM25 function object
        """
        return Function(
            name=self.config.bm25_function_name,
            function_type=FunctionType.BM25,
            input_field_names=[self.config.content_field],
            output_field_names=[self.config.sparse_vector_field],
        )

    def define_indexes(self) -> Any:
        """Define indexes for the Milvus collection.

        Returns:
            Any: Milvus index parameters object
        """
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=self.config.sparse_vector_field,
            index_type=self.config.sparse_index_type,
            metric_type=self.config.sparse_metric_type,
        )
        index_params.add_index(
            field_name=self.config.dense_vector_field,
            index_type=self.config.dense_index_type,
            metric_type=self.config.dense_metric_type,
        )
        return index_params

    def create_collection(self) -> None:
        """Create the Milvus collection with schema and indexes."""
        # Drop collection if exists
        if self.client.has_collection(self.config.collection_name):
            self.client.drop_collection(self.config.collection_name)
            logger.info(f"Dropped existing collection '{self.config.collection_name}'")

        # Define schema and indexes
        schema = self.define_schema()
        bm25_function = self.define_bm25_function()
        schema.add_function(bm25_function)
        index_params = self.define_indexes()

        # Create the collection
        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection '{self.config.collection_name}' created successfully")

    def setup_model(self) -> None:
        """Set up the sentence transformer model."""
        self.model = SentenceTransformer(
            self.config.model_name,
            tokenizer_kwargs={"padding_side": self.config.padding_side},
        )
        logger.info(f"Model '{self.config.model_name}' loaded successfully")

    def start_multiprocess_pool(self) -> None:
        """Start multi-process pool for embedding generation."""
        logger.info("Starting multi-process pool for embedding generation...")
        self.pool = self.model.start_multi_process_pool(
            target_devices=self.config.target_devices
        )

    def stop_multiprocess_pool(self) -> None:
        """Stop multi-process pool."""
        if self.pool:
            logger.info("Stopping multi-process pool...")
            self.model.stop_multi_process_pool(self.pool)

    def load_dataset(self) -> Any:
        """Load the dataset from HuggingFace.

        Returns:
            Any: HuggingFace dataset
        """
        logger.info("Loading dataset from HuggingFace...")
        dataset = load_dataset(
            self.config.hf_dataset_name, split=self.config.dataset_split
        )
        logger.info(f"Loaded {len(dataset)} documents")
        return dataset

    def prepare_entities(
        self, batch_dict: dict[str, list], embeddings: list[list[float]]
    ) -> list[dict[str, Any]]:
        """Prepare entities for insertion into Milvus.

        Args:
            batch_dict (Dict[str, List]): Batch of documents
            embeddings (List[List[float]]): Embeddings for the documents

        Returns:
            List[Dict[str, Any]]: List of entities ready for insertion
        """
        entities = []
        batch_size = len(batch_dict[self.config.id_field])

        for j in range(batch_size):
            entity = {
                self.config.id_field: batch_dict[self.config.id_field][j],
                self.config.file_name_field: batch_dict[self.config.file_name_field][j],
                self.config.content_field: batch_dict[self.config.content_field][j],
                self.config.dense_vector_field: embeddings[j],
                self.config.metadata_field: {
                    "file_name": batch_dict[self.config.file_name_field][j]
                },
            }
            entities.append(entity)

        return entities

    def insert_batch(
        self, entities: list[dict[str, Any]], batch_index: int, total_batches: int
    ) -> None:
        """Insert a batch of entities into Milvus.

        Args:
            entities (List[Dict[str, Any]]): Entities to insert
            batch_index (int): Current batch index
            total_batches (int): Total number of batches
        """
        try:
            self.client.insert(self.config.collection_name, entities)
            logger.debug(f"Inserted batch {batch_index + 1}/{total_batches}")
        except Exception as e:
            logger.error(f"Error inserting batch {batch_index + 1}: {e}")
            raise

    def index_documents(self) -> int:
        """Index all documents from the dataset into Milvus.

        Returns:
            int: Total number of inserted documents
        """
        # Load dataset
        dataset = self.load_dataset()

        # Process in batches
        total_inserted = 0
        total_batches = (len(dataset) - 1) // self.config.batch_size + 1

        for i in tqdm.tqdm(
            range(0, len(dataset), self.config.batch_size), desc="Indexing documents"
        ):
            batch_dict = dataset[i : i + self.config.batch_size]

            # Generate embeddings using multi-process pool
            texts = batch_dict[self.config.content_field]
            embeddings = self.model.encode(
                texts,
                pool=self.pool,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # Prepare entities for insertion
            entities = self.prepare_entities(batch_dict, embeddings)

            # Insert data
            self.insert_batch(entities, i // self.config.batch_size, total_batches)
            total_inserted += len(entities)

        logger.info(f"Successfully inserted {total_inserted} documents")
        return total_inserted
