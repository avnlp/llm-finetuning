from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, notebook_login


# Configuration
DATASET_NAME = "mandarjoshi/trivia_qa"
CONFIG = "rc"  # Reading comprehension config
SAMPLE_SIZE = 1000
TEST_SIZE = 0.1
VAL_SIZE = 0.1
HF_REPO_ID = "your-hf-username/triviaqa-processed"  # Change to your username
FORMATTING_TEMPLATE = """<|start_header_id|>user<|end_header_id|>\n\n
Answer the question based on the context below. Keep your response concise.

Question: {question}

Context:
{context}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"""


def format_context(example):
    """Format context from entity pages and search results."""
    context_text = ""

    # Add entity page context
    if example["entity_pages"]:
        page = example["entity_pages"][0]
        context_text += f"Title: {page['title']}\n"
        context_text += f"{page['wiki_context']}\n\n"

    # Add search results context
    if example["search_results"]:
        result = example["search_results"][0]
        context_text += f"Search Result: {result['title']}\n"
        context_text += f"{result['search_context']}\n\n"

    return context_text.strip()


def preprocess_triviaqa(example):
    """Preprocess each example into the required format."""
    # Extract and format context
    context_str = format_context(example)

    return {
        "id": example["question_id"],
        "question": example["question"],
        "answer": example["value"],
        "context": context_str,
        "aliases": ", ".join(example["aliases"]),
        "question_source": example["question_source"],
        "type": example["type"],
        "text": FORMATTING_TEMPLATE.format(
            question=example["question"], context=context_str, answer=example["value"]
        ),
    }


def create_dataset():
    """Create and split the dataset."""
    # Load and sample dataset
    full_dataset = load_dataset(DATASET_NAME, CONFIG, split="train")
    sampled = full_dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

    # Preprocess
    processed = sampled.map(preprocess_triviaqa)

    # Convert to pandas for easy splitting
    df = processed.to_pandas()

    # Create splits
    test_val_size = int(SAMPLE_SIZE * (TEST_SIZE + VAL_SIZE))
    train_df, test_val_df = df[:-test_val_size], df[-test_val_size:]
    test_size = int(test_val_size * TEST_SIZE / (TEST_SIZE + VAL_SIZE))
    test_df, val_df = test_val_df[:test_size], test_val_df[test_size:]

    # Create DatasetDict
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
            "validation": Dataset.from_pandas(val_df),
        }
    )


def upload_to_hf(dataset_dict, repo_id):
    """Upload dataset to Hugging Face Hub."""
    HfApi()
    dataset_dict.push_to_hub(repo_id)
    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    # Login to HF Hub
    notebook_login()

    # Create dataset
    print("Creating dataset...")
    trivia_ds = create_dataset()

    # Print sample
    print("\nSample training example:")
    print(trivia_ds["train"][0]["text"])

    # Upload to Hub
    print(f"\nUploading dataset to {HF_REPO_ID}...")
    upload_to_hf(trivia_ds, HF_REPO_ID)

    print("\nProcess completed successfully!")
    print(
        f"Dataset sizes: Train={len(trivia_ds['train'])}, "
        f"Val={len(trivia_ds['validation'])}, Test={len(trivia_ds['test'])}"
    )
