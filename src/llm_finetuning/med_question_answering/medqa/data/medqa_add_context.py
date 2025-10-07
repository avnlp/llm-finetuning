from pathlib import Path

import yaml
from config import MedQAContextAdditionConfig
from medqa_context_adder import MedQAContextAdder


def main(config_path: str):
    """Run the MedQA context addition process."""
    config_file = Path(config_path)

    if not config_file.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_file) as file:
        config_dict = yaml.safe_load(file)

    config = MedQAContextAdditionConfig(**config_dict)

    # Create context adder and process
    context_adder = MedQAContextAdder(config)

    # Connect to Milvus and set up model
    context_adder.connect_to_milvus()
    context_adder.setup_model()

    # Load dataset
    dataset_dict = context_adder.load_dataset()

    # Add context columns
    updated_dataset_dict = context_adder.add_context_to_dataset(dataset_dict)

    # Upload to HuggingFace Hub
    context_adder.upload_to_huggingface(updated_dataset_dict)

    print("Processing completed successfully!")


if __name__ == "__main__":
    main("medqa_context_addition_config.yaml")
