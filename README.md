# Document Ranking Models

This repository contains the implementation and evaluation of three Information Retrieval (IR) models designed to rank documents in specific datasets according to their corresponding topics.

## Models Implemented
1. **Baseline 1:** BM25 Model.
2. **Baseline 2:** Jelinek-Mercer (JM) Smoothed Language Model.
3. **Model C (Custom):** A ranking model leveraging Pseudo Relevance Feedback (PRF) and Rocchio-based query expansion.

## Prerequisites and Setup
The project requires the following Python libraries[cite: 18]:
* `os`, `sys`, `math`, `re` 
* `xml.etree.ElementTree` (for XML parsing) 
* `nltk`, `nltk.stem` (for NLP preprocessing: tokenization, stop-word removal, and stemming) 
* `scipy` (for significance testing) 
* `pandas` 

### Installation Steps
1. Create a virtual environment: `python -m venv venv` [cite: 47]
2. Activate the virtual environment:
   * Windows: `.venv\Scripts\activate` 
   * macOS/Linux: `source .venv/bin/activate` 
3. Install the required packages: `pip install nltk pandas scipy` 

## Project Structure
  * `Doc Collection/` - Contains the XML document datasets.
  * `Relevant Judgements/` - Contains the relevance judgement text files.
  * `ModelOutputs/` - Directory where the `.dat` ranking output files are generated.
  * `eval/` - Directory where evaluation reports and t-test results are saved.
  * `Topics.txt` - Contains the queries (topic IDs and titles).
  * `A2_solution.py` - The main retrieval program.
  * `Evaluation.py` - The evaluation and significance testing script.

## Execution
**1. Run the Retrieval Models:**
```bash
python A2_solution.py
