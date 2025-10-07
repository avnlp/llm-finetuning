from datasets import DatasetDict, load_dataset


# Load the TriviaQA dataset (rc subset)
print("Loading TriviaQA dataset...")
dataset = load_dataset("mandarjoshi/trivia_qa", "rc.web")


def format_dataset(example):
    example["context"] = " ".join(
        "\n".join(example["search_results"]["search_context"]).split("\n")
    )
    example["answer"] = example["answer"]["normalized_value"]
    return example


# Add context and answer to the dataset
train_dataset = dataset["train"].map(format_dataset, num_proc=4)
test_dataset = dataset["test"].map(format_dataset, num_proc=4)
validation_dataset = dataset["validation"].map(format_dataset, num_proc=4)

# Filter non-empty context
train_filtered_dataset = train_dataset.filter(lambda x: len(x["context"]) > 0)
test_filtered_dataset = test_dataset.filter(lambda x: len(x["context"]) > 0)
validation_filtered_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)

# Filter out examples where context + question is less than 4096 characters
train_filtered_dataset_context = train_filtered_dataset.filter(
    lambda x: (len(x["question"]) + len(x["context"])) < 4096
)
test_filtered_dataset_context = test_filtered_dataset.filter(
    lambda x: (len(x["question"]) + len(x["context"])) < 4096
)
validation_filtered_dataset_context = validation_filtered_dataset.filter(
    lambda x: (len(x["question"]) + len(x["context"])) < 4096
)

print(train_filtered_dataset_context.shape)
print(test_filtered_dataset_context.shape)
print(validation_filtered_dataset_context.shape)

train_filtered_dataset_context = train_filtered_dataset_context.remove_columns(
    ["question_source", "search_results", "entity_pages"]
)
test_filtered_dataset_context = test_filtered_dataset_context.remove_columns(
    ["question_source", "search_results", "entity_pages"]
)
validation_filtered_dataset_context = (
    validation_filtered_dataset_context.remove_columns(
        ["question_source", "search_results", "entity_pages"]
    )
)


# Create a new dataset dictionary with the subsets
subset_dataset = DatasetDict(
    {
        "train": train_filtered_dataset_context,
        "test": test_filtered_dataset_context,
        "validation": validation_filtered_dataset_context,
    }
)

# from huggingface_hub import notebook_login
# notebook_login()

DATASET_NAME = "awinml/triviaqa_processed"
HF_TOKEN = ("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",)

# Print dataset information
print("\nSubset dataset structure:")
for split_name, split_data in subset_dataset.items():
    print(f"{split_name}: {len(split_data)} examples")

print(f"\nPushing dataset to {DATASET_NAME}...")
subset_dataset.push_to_hub(DATASET_NAME, private=False)
print(f"Successfully pushed to https://huggingface.co/datasets/{DATASET_NAME}")
