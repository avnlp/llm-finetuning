# MedQA

Paper: [What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams](https://arxiv.org/abs/2009.13081)
Github: [https://github.com/jind11/MedQA](https://github.com/jind11/MedQA)

## Data

The data can be downloaded from: [Google Drive link by author](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view).

The data is in the following format:

- Questions
  - Mainland
  - Taiwan
  - US
- Textbooks
  - English
  - Simplified Chinese (Paragraph Splitting)
  - Simplified Chinese (Sentence Splitting)

### Questions

- For Question-Answers, the data has three sources: US, Mainland of China, and Taiwan District, which are put in folders, respectively.
- All files for QAs are in jsonl file format, where each line is a data sample as a dict.
- There is a "qbank" file that contains all data samples. The official random split into train, dev, and test sets are also provided.
- The data also has the "4_options" version where the options for each question has been reduced to 4.
- The "metamap" folders are extracted medical related phrases using the Metamap tool.

### Corpus

- The corpus of textbooks has two languages: English and simplified Chinese.
- For simplified Chinese, the corpus is available in two kinds of sentence splitting: one is split by sentences, and the other is split by paragraphs.
