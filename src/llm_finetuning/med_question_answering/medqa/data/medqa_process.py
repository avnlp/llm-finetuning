"""Process MedQA dataset files."""

from pathlib import Path

import yaml
from config import MedQADatasetProcessingConfig  # type: ignore[import-not-found]
from medqa_dataset_processor import (  # type: ignore[import-not-found]
    MedQADatasetProcessor,  # type: ignore[import-not-found]
)


def main(config_path: str) -> None:
    """Process the MedQA dataset."""
    config_file = Path(config_path)

    if not config_file.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_file) as file:
        config_dict = yaml.safe_load(file)
    config = MedQADatasetProcessingConfig(**config_dict)

    # Create processor and process
    processor = MedQADatasetProcessor(config)
    # Load all data
    data = processor.load_all_data()

    # Create datasets
    dataset_dict = processor.create_datasets(data)

    # Print dataset info
    processor.print_dataset_info(dataset_dict)

    # Upload to HuggingFace Hub
    processor.upload_to_huggingface(dataset_dict)


if __name__ == "__main__":
    main("medqa_dataset_processing_config.yaml")
