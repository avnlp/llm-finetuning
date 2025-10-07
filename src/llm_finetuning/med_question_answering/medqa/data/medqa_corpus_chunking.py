from pathlib import Path

import yaml
from config import MedQACorpusChunkingConfig
from medqa_corpus_chunker import MedQACorpusChunker


def main(config_path: str):
    """Run the MedQA corpus chunking process."""
    # Load configuration
    config_file = Path(config_path)

    if not config_file.exists():
        msg = f"Configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_file) as file:
        config_dict = yaml.safe_load(file)

    config = MedQACorpusChunkingConfig(**config_dict)

    # Create chunker and process
    chunker = MedQACorpusChunker(config)

    # Chunk files
    chunks = chunker.chunk_files()

    # Save chunks to JSONL
    chunker.save_chunks_to_jsonl(chunks)

    # Create HuggingFace dataset
    dataset_dict = chunker.create_hf_dataset(chunks)

    # Save HuggingFace dataset locally
    chunker.save_hf_dataset_locally(dataset_dict)

    # Push to HuggingFace Hub
    dataset_dict.push_to_hub(config.hf_dataset_name, private=config.hf_private)
    print("Processing completed successfully!")


if __name__ == "__main__":
    main("medqa_corpus_chunker_config.yaml")
