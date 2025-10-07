import json
import logging
from typing import Any

import tqdm
from config import MedQAContextAdditionConfig
from datasets import Dataset, DatasetDict, load_dataset
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedQAContextAdder:
    """A class to add context columns to the MedQA dataset."""

    def __init__(self, config: MedQAContextAdditionConfig):
        """Initialize the MedQAContextAdder with configuration.

        Args:
            config (MedQAContextAdditionConfig): Configuration object
        """
        self.config = config
        self.client = None
        self.model = None

    def connect_to_milvus(self) -> None:
        """Connect to Milvus database."""
        self.client = MilvusClient(
            uri=self.config.milvus_uri, token=self.config.milvus_token
        )
        logger.info("Connected to Milvus successfully")

    def setup_model(self) -> None:
        """Set up the sentence transformer model."""
        self.model = SentenceTransformer(
            self.config.model_name,
            tokenizer_kwargs={"padding_side": self.config.padding_side},
        )
        logger.info(f"Model '{self.config.model_name}' loaded successfully")

    def load_dataset(self) -> DatasetDict:
        """Load the MedQA dataset from HuggingFace.

        Returns:
            DatasetDict: HuggingFace dataset dictionary
        """
        logger.info("Loading MedQA dataset...")
        return load_dataset(self.config.input_hf_dataset_name)

    def format_context_results(
        self, results: list[dict], top_k: int = 10
    ) -> dict[str, dict[str, Any]]:
        """Format retrieval results into the required context format.

        Args:
            results (List[Dict]): Retrieval results
            top_k (int): Number of results to include

        Returns:
            Dict[str, Dict[str, Any]]: Formatted context
        """
        context = {}
        for i, result in enumerate(results[:top_k]):
            context[str(i + 1)] = {
                "content": result["entity"]["content"],
                "metadata": result["entity"]["metadata"],
            }
        return context

    def retrieve_context_bm25(
        self, queries: list[str]
    ) -> list[dict[str, dict[str, Any]]]:
        """Retrieve context using BM25 (sparse search).

        Args:
            queries (List[str]): List of queries

        Returns:
            List[Dict[str, Dict[str, Any]]]: List of context dictionaries
        """
        contexts = []
        for query in tqdm.tqdm(queries, desc="BM25 Retrieval"):
            try:
                results = self.client.search(
                    collection_name=self.config.collection_name,
                    data=[query],
                    anns_field="sparse_vector",
                    limit=self.config.top_k,
                    output_fields=["content", "metadata"],
                )
                context = self.format_context_results(results[0], self.config.top_k)
                contexts.append(context)
            except Exception as e:
                logger.error(
                    f"Error in BM25 retrieval for query '{query[:50]}...': {e}"
                )
                contexts.append({})
        return contexts

    def retrieve_context_dense(
        self, queries: list[str]
    ) -> list[dict[str, dict[str, Any]]]:
        """Retrieve context using dense embeddings.

        Args:
            queries (List[str]): List of queries

        Returns:
            List[Dict[str, Dict[str, Any]]]: List of context dictionaries
        """
        contexts = []
        try:
            # Generate embeddings for all queries at once
            query_embeddings = self.model.encode(
                queries, batch_size=self.config.batch_size, show_progress_bar=True
            )

            for query_embedding in tqdm.tqdm(query_embeddings, desc="Dense Retrieval"):
                try:
                    results = self.client.search(
                        collection_name=self.config.collection_name,
                        data=[query_embedding],
                        anns_field="dense_vector",
                        limit=self.config.top_k,
                        output_fields=["content", "metadata"],
                    )
                    context = self.format_context_results(results[0], self.config.top_k)
                    contexts.append(context)
                except Exception as e:
                    logger.error(f"Error in dense retrieval: {e}")
                    contexts.append({})
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty contexts for all queries if embedding generation fails
            contexts = [{}] * len(queries)
        return contexts

    def add_context_to_split(self, dataset: Dataset, split_name: str) -> Dataset:
        """Add context columns to a dataset split.

        Args:
            dataset (Dataset): Dataset split
            split_name (str): Name of the split

        Returns:
            Dataset: Updated dataset with context columns
        """
        logger.info(f"Processing {split_name} split with {len(dataset)} examples")

        # Process in batches to avoid memory issues
        updated_data = []

        for i in tqdm.tqdm(
            range(0, len(dataset), self.config.batch_size),
            desc=f"Processing {split_name}",
        ):
            batch_dict = dataset[i : i + self.config.batch_size]
            batch_size = len(batch_dict[self.config.question_field])

            # Extract queries
            queries = batch_dict[self.config.question_field]

            # Retrieve contexts
            context_bm25_list = self.retrieve_context_bm25(queries)
            context_dense_list = self.retrieve_context_dense(queries)

            # Add context columns to each example in the batch
            for j in range(batch_size):
                # Create example with all original fields
                example = {key: batch_dict[key][j] for key in batch_dict}
                # Add new context fields as JSON strings
                example[self.config.context_bm25_field] = json.dumps(
                    context_bm25_list[j]
                )
                example[self.config.context_qwen_embed_field] = json.dumps(
                    context_dense_list[j]
                )
                updated_data.append(example)

        # Create updated dataset
        return Dataset.from_list(updated_data)

    def add_context_to_dataset(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Add context columns to all splits of the dataset.

        Args:
            dataset_dict (DatasetDict): Original dataset dictionary

        Returns:
            DatasetDict: Updated dataset dictionary with context columns
        """
        updated_splits = {}

        for split_name, dataset in dataset_dict.items():
            updated_splits[split_name] = self.add_context_to_split(dataset, split_name)

        return DatasetDict(updated_splits)

    def upload_to_huggingface(self, dataset_dict: DatasetDict) -> None:
        """Upload dataset to HuggingFace Hub.

        Args:
            dataset_dict (DatasetDict): Dataset dictionary to upload
        """
        logger.info("Uploading updated dataset to Hugging Face...")
        dataset_dict.push_to_hub(
            self.config.output_hf_dataset_name, private=self.config.hf_private
        )

    def process(self, upload_to_hub: bool = True) -> DatasetDict:
        """Main processing method to add context and upload the dataset.

        Args:
            upload_to_hub (bool): Whether to upload to HuggingFace Hub

        Returns:
            DatasetDict: Processed dataset dictionary
        """
        return updated_dataset_dict
