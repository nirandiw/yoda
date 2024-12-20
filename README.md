# Chatbot Groundedness Guardrail System

This repository demonstrates a solution for designing and implementing a guardrail system for ensuring the groundedness of chatbot responses. The system is tailored to assess whether chatbot responses about bank retail products are accurate, relevant, and policy-compliant.  

## Project Overview

The chatbot provides information on **retail banking products** (e.g., credit cards, home loans, savings accounts). It uses a **Large Language Model (LLM)** as the core of its conversational capabilities. The guardrail system ensures that chatbot responses are:
- Factually accurate and derived from the provided context.
- Relevant to the user's question.
- Compliant with company policies.

### Key Features
1. **Binary Groundedness Evaluation**  
   Uses an external API or pre-trained models to determine whether a response is grounded (`SAFE`) or ungrounded (`UNSAFE`).

2. **Probabilistic Groundedness Evaluation**  
   Employs a probabilistic model to provide a risk score for responses based on token probabilities.

3. **Custom Evaluation with Scoring**  
   Breaks down each response into sentences and evaluates:
   - Factual accuracy.
   - Relevance to the query.
   - Policy compliance.
   - Contextual coherence.

4. **Synthetic Test Cases**  
   Includes scripts for generating test cases to validate both grounded and ungrounded responses.

## Usage
Install the necessary libs
```
pip install -r requirements.txt
```

Call the function. 
```
import guardrail_groundedness from yoda_custom.py
from auth import get_openai_client

client = get_openai_client()
choices = guardrail_groundedness(context_text,query, response_text, client )
logger.info(f"Grounding information {choices[0].message.content.strip('`json')}")
response_json = choices[0].message.content.strip('`json')
response_data = json.loads(response_json)
for item in response_data:
   groundedness = is_sentence_grounded(item)
   logger.info(f" Sentence: {item['sentence']}... Grounded??: {groundedness}")

```
---

## Repository Structure

```plaintext
├──data # Input pds files
├──output # synthetic data generated
├──src
    ├── yoda_ootb.py       # Handles out-of-the-box (OOTB) model evaluations using the GRANITE API
    ├── yoda_custom.py     # Custom GPT-based evaluation for groundedness
    ├── utils.py           # Utility functions for token parsing, model interactions, and file handling
    ├── prompts.py         # Prompt templates for LLM interactions and evaluations
└── README.md          # Documentation for the repository
