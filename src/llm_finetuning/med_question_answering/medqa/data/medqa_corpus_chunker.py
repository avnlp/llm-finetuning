import json
import os
from pathlib import Path

import tqdm
from config import MedQACorpusChunkingConfig
from datasets import Dataset, DatasetDict
from unstructured.partition.text import partition_text


class MedQACorpusChunker:
    """A class to chunk MedQA corpus files into smaller pieces for processing."""

    def __init__(self, config: MedQACorpusChunkingConfig):
        """Initialize the MedQACorpusChunker with configuration.

        Args:
            config (MedQACorpusChunkingConfig): Configuration object
        """
        self.config = config

    def get_text_files(self) -> list[str]:
        """Get all text files in the corpus directory.

        Returns:
            list[str]: List of text file names
        """
        txt_files = [
            f
            for f in os.listdir(self.config.corpus_dir)
            if f.endswith(self.config.file_extension)
        ]
        return sorted(txt_files)

    def chunk_files(self) -> list[dict]:
        """Process all files and create chunks.

        Returns:
            list[dict]: List of chunk dictionaries
        """
        txt_files = self.get_text_files()
        print(f"Found {len(txt_files)} text files to process")

        all_chunks = []
        for fname in tqdm.tqdm(txt_files):
            fpath = os.path.join(self.config.corpus_dir, fname)

            # Partition text by title using unstructured
            elements = partition_text(
                filename=fpath, partition_by=self.config.partition_by
            )

            # Create chunks from the partitioned elements
            for i, element in enumerate(elements):
                chunk_data = {
                    "id": f"{Path(fname).stem}_{i}",
                    "file_name": fname,
                    "content": element.text,
                }
                all_chunks.append(chunk_data)

        print(f"Created {len(all_chunks)} chunks total")
        return all_chunks

    def save_chunks_to_jsonl(self, chunks: list[dict]) -> None:
        """Save all chunks to a single JSONL file.

        Args:
            chunks (list[dict]): List of chunk dictionaries
        """
        with open(self.config.chunked_dataset_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        print(f"Combined dataset saved to '{self.config.chunked_dataset_file}'")

    def create_hf_dataset(self, chunks: list[dict]) -> DatasetDict:
        """Create HuggingFace dataset from the chunks.

        Args:
            chunks (list[dict]): List of chunk dictionaries

        Returns:
            DatasetDict: HuggingFace dataset dictionary
        """
        # Create HuggingFace dataset from the combined data
        dataset = Dataset.from_list(chunks)

        # Create dataset dict with a single train split
        return DatasetDict({"train": dataset})

    def save_hf_dataset_locally(self, dataset_dict: DatasetDict) -> None:
        """Save HuggingFace dataset locally.

        Args:
            dataset_dict (DatasetDict): HuggingFace dataset dictionary
        """
        dataset_dict.save_to_disk(self.config.hf_dataset_dir)
        print(f"Dataset saved locally to '{self.config.hf_dataset_dir}'")

    def push_to_hf_hub(self, dataset: Dataset) -> None:
        """Push dataset to HuggingFace Hub.

        Args:
            dataset (Dataset): HuggingFace dataset
        """
        dataset.push_to_hub(self.config.hf_dataset_name, private=self.config.hf_private)
