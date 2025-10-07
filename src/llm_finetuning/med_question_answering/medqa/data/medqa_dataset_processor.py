import json
from typing import Any

from config import MedQADatasetProcessingConfig
from datasets import Dataset, DatasetDict


class MedQADatasetProcessor:
    """A class to process MedQA dataset files and upload to HuggingFace."""

    def __init__(self, config: MedQADatasetProcessingConfig):
        """Initialize the MedQADatasetProcessor with configuration.

        Args:
            config (MedQADatasetProcessingConfig): Configuration object
        """
        self.config = config

    def load_jsonl_file(self, file_path: str) -> list[dict[str, Any]]:
        """Load data from a JSONL file.

        Args:
            file_path (str): Path to the JSONL file

        Returns:
            list[dict[str, Any]]: List of data dictionaries
        """
        data = []
        with open(file_path) as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data

    def load_all_data(self) -> dict[str, list[dict[str, Any]]]:
        """Load data from all split files.

        Returns:
            dict[str, list[dict[str, Any]]]: Dictionary with data for each split
        """
        data = {}
        data[self.config.train_split_name] = self.load_jsonl_file(
            self.config.train_file
        )
        data[self.config.validation_split_name] = self.load_jsonl_file(
            self.config.dev_file
        )
        data[self.config.test_split_name] = self.load_jsonl_file(self.config.test_file)
        return data

    def create_datasets(self, data: dict[str, list[dict[str, Any]]]) -> DatasetDict:
        """Convert data to Dataset objects.

        Args:
            data (dict[str, list[dict[str, Any]]]): Data for each split

        Returns:
            DatasetDict: HuggingFace dataset dictionary
        """
        # Convert to Dataset objects
        train_dataset = Dataset.from_list(data[self.config.train_split_name])
        dev_dataset = Dataset.from_list(data[self.config.validation_split_name])
        test_dataset = Dataset.from_list(data[self.config.test_split_name])

        # Create a DatasetDict
        return DatasetDict(
            {
                self.config.train_split_name: train_dataset,
                self.config.validation_split_name: dev_dataset,
                self.config.test_split_name: test_dataset,
            }
        )

    def print_dataset_info(self, dataset_dict: DatasetDict) -> None:
        """Print information about the dataset.

        Args:
            dataset_dict (DatasetDict): HuggingFace dataset dictionary
        """
        train_dataset = dataset_dict[self.config.train_split_name]
        dev_dataset = dataset_dict[self.config.validation_split_name]
        test_dataset = dataset_dict[self.config.test_split_name]

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(dev_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        print("\nExample from train dataset:")
        print(train_dataset[0])

    def upload_to_huggingface(self, dataset_dict: DatasetDict) -> None:
        """Upload dataset to HuggingFace Hub.

        Args:
            dataset_dict (DatasetDict): HuggingFace dataset dictionary
        """
        dataset_dict.push_to_hub(
            self.config.hf_dataset_name, private=self.config.hf_private
        )
