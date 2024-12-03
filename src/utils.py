import torch
import math
from PyPDF2 import PdfReader
import logging
import json
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IBM_GRANITE = "ibm-granite/granite-guardian-3.0-8b"
GPT_MODEL = "gpt-4o"
SAFE_TOKEN = "No"
UNSAFE_TOKEN = "Yes"
N_LOGS_PROBS = 20

"""Constructs a message payload for GPT OOTB (out-of-the-box)."""
def get_message_gg_ootb(context, response_text, risk):
    message =  [{
        "role": "system",
        "content": risk
    }, 
    {"role": "context",
     "content": context
    }
    , 
    {"role": "assistant",
     "content": response_text
    }
    ]
    return message

def get_message_gg_local(context, response_text, risk):
    messages = [{"role": "context", "content": context}, {"role": "assistant", "content": response_text}]
    guardian_config = {"risk_name": risk}
    return messages, guardian_config


def parse_output(output, input_len, tokenizer):
    label, prob_of_risk = None, None
    if N_LOGS_PROBS > 0:

        list_index_logprobs_i = [torch.topk(token_i, k=N_LOGS_PROBS, largest=True, sorted=True)
                                 for token_i in list(output.scores)[:-1]]
        if list_index_logprobs_i is not None:
            prob = get_probablities(list_index_logprobs_i, tokenizer)
            prob_of_risk = prob[1]

    res = tokenizer.decode(output.sequences[:,input_len:][0],skip_special_tokens=True).strip()
    if UNSAFE_TOKEN.lower() == res.lower():
        label = UNSAFE_TOKEN
    elif SAFE_TOKEN.lower() == res.lower():
        label = SAFE_TOKEN
    else:
        label = "Failed"

    return label, prob_of_risk.item()

def get_probablities(logprobs, tokenizer):
    safe_token_prob = 1e-50
    unsafe_token_prob = 1e-50
    for gen_token_i in logprobs:
        for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
            decoded_token = tokenizer.convert_ids_to_tokens(index)
            if decoded_token.strip().lower() == SAFE_TOKEN.lower():
                safe_token_prob += math.exp(logprob)
            if decoded_token.strip().lower() == UNSAFE_TOKEN.lower():
                unsafe_token_prob += math.exp(logprob)

    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
    )

    return probabilities

def load_pds(pdf_path='../data/pet-insurance-combined-fsg-pds.pdf'):

    reader = PdfReader(pdf_path)
    text_list =[]
    # Extract text from all pages
    for page in reader.pages:
        text_list.append(page.extract_text())
    
    return text_list

def save_json_to_file(data, file_path):
    """Utility function to save JSON data to a file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    logger.info(f"Saved JSON to {file_path}")