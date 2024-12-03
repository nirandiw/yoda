from auth import get_openai_client
from utils import load_pds, GPT_MODEL, save_json_to_file
from prompts import test_true_generation, test_false_generation
from yoda_custom import guardrail_groundedness, is_sentence_grounded, is_answer_grounded
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global paths for outputs
OUTPUT_DIR = "../output"
POSITIVE_TEST_PATH = f"{OUTPUT_DIR}/p_test.json"
NEGATIVE_TEST_PATH = f"{OUTPUT_DIR}/n_test.json"


def generate_test_set(client, prompt, output_path, domain=None):
    """Generate a test set based on the provided prompt."""
    if domain:
        knowledge_base = load_pds()
        prompt = prompt.format(pds=knowledge_base, domain=domain)

    messages = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.7, n=10
    )

    response_json = json.loads(response.choices[0].message.content.strip('`json'))
    save_json_to_file(response_json, output_path)
    return response_json

def generate_positive_test(client, domain="insurance"):
    """Generate positive test cases."""
    return generate_test_set(client, test_true_generation, POSITIVE_TEST_PATH, domain)

def generate_negative_test(client):
    """Generate negative test cases."""
    return generate_test_set(client, test_false_generation, NEGATIVE_TEST_PATH)

def process_row(row, client):
    """Process a single row to evaluate groundedness."""
    context, query, response_text = row['context'], row['query'], row['response']
    choices = guardrail_groundedness(context, query, response_text, client)
    response_json = choices[0].message.content.strip('`json')
    response_data = json.loads(response_json)
    result_set = []
    for item in response_data:
        groundedness = is_sentence_grounded(item)
        result_set.append({
            "factual accuracy": item.get("factualAccuracy", 0),
            "relevancy": item.get("relevance", 0),
            "policy compliance": item.get("policyCompliance", 0),
            "contextual coherence": item.get("contextualCoherence", 0),
            "groundedness": groundedness,
            "genai response": item.get("sentence", ""),
        })
    answer_grounded=is_answer_grounded(response_data)
    result_set.append({"answer_grounded":answer_grounded})
    logger.info("Completed processing row")
    return result_set, row



def evaluate(client, eval_set):
    """Evaluate the dataset for groundedness."""
    match_count = 0
    full_count=0
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_row, row, client): row
            for _, row in eval_set.iterrows()
        }
        for future in as_completed(futures):
            try:
                results, data = future.result()
                full_count +=1
                is_match = results[-1]["answer_grounded"] == data["grounded"]
                match_count += is_match
                logger.info(f"Match: {is_match}, Total Matches %: {np.divide(match_count, full_count)}")
            except Exception as e:
                logger.error(f"Error processing row: {e}")
    logger.info(f"Evaluation completed. Total matches: {match_count}")

if __name__ == "__main__":
    client = get_openai_client()
    df = pd.read_json(f"{OUTPUT_DIR}/test.json", orient="columns")
    # generate_positive_test(client)
    # generate_negative_test(client)
    evaluate(client, df)
