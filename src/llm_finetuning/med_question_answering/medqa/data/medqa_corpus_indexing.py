from pathlib import Path

import yaml
from config import MedQAIndexingConfig
from medqa_indexer import MedQAIndexer


def main(config_path: str):
    """Run the MedQA corpus indexing process."""
    # Load configuration
    config_file = Path(config_path)

    if not config_file.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_file) as file:
        config_dict = yaml.safe_load(file)

    config = MedQAIndexingConfig(**config_dict)

    # Create indexer and process
    indexer = MedQAIndexer(config)

    # Connect to Milvus
    indexer.connect_to_milvus()

    # Create collection
    indexer.create_collection()

    # Set up model and multiprocessing
    indexer.setup_model()
    indexer.start_multiprocess_pool()

    try:
        # Index documents
        total_inserted = indexer.index_documents()
        print(
            f"Processing completed successfully! Inserted {total_inserted} documents."
        )
    finally:
        # Clean up multiprocessing resources
        indexer.stop_multiprocess_pool()


if __name__ == "__main__":
    main("medqa_indexing_config.yaml")
