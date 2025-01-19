import datasets
import jsonlines

# Task instructions for the dataset prefix
TASK_INST = {
    "earnings-calls": "Answer the question based on the given earnings call transcript."
}

# Parameters
output_file = "earnings_calls_training_1000.jsonl"
data_prefix = "earnings-calls"

# Load the dataset
data = datasets.load_dataset("lamini/earnings-calls-qa")["train"]

# Process the dataset
new_data = []
for idx, item in enumerate(data):
    # Extract fields
    question = item["question"]
    transcript = item["transcript"]
    answer = item["answer"]

    # Format question and instruction
    input_context = f"Transcript:\n{transcript}\n\nQuestion:\n{question}"
    instruction = f"{TASK_INST[data_prefix]} ## Input:\n\n{input_context}"

    # Append to the dataset
    new_data.append(
        {
            "instruction": instruction,
            "output": answer,
            "input": "",
            "id": f"{data_prefix}_{idx}",  # Generate a unique ID using the index
            "dataset_name": data_prefix,
        }
    )

# Save to a JSONL file
with jsonlines.open(output_file, "w") as writer:
    writer.write_all(new_data)

print(f"Generated training data for the first 1000 rows saved to {output_file}")
