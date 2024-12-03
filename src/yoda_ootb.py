from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import json
import logging
from utils import get_message_gg_ootb, get_message_gg_local, parse_output, IBM_GRANITE

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "granite3-guardian:8b"
GRANITE_API_URL = "http://0.0.0.0:11435/api/chat"
MAX_TOKENS = 20

# Groundedness Check
def check_groundedness(model, tokenizer, messages, guardian_config):
    """Check groundedness of a response using a model and tokenizer."""
    logger.info("Preparing input IDs for groundedness check.")
    input_ids = tokenizer.apply_chat_template(
        messages, 
        guardian_config=guardian_config, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    input_len = input_ids.shape[1]

    logger.info("Generating output from model.")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=MAX_TOKENS,
            return_dict_in_generate=True,
            output_scores=True,
        )

    label, prob_of_risk = parse_output(output, input_len, tokenizer)
    logger.info(f"Groundedness check complete. Label: {label}, Probability of risk: {prob_of_risk}")
    return label, prob_of_risk


def grounding_with_probability(context, response_text, risk="groundedness"):
    """Evaluate grounding with probabilities using a pre-trained model."""
    logger.info("Loading model and tokenizer for grounding check.")
    model = AutoModelForCausalLM.from_pretrained(IBM_GRANITE)
    tokenizer = AutoTokenizer.from_pretrained(IBM_GRANITE)
    
    model = model.to(torch.device("cpu")).eval()
    messages, guardian_config = get_message_gg_local(context, response_text, risk)

    logger.info("Starting probabilistic grounding evaluation.")
    return check_groundedness(model, tokenizer, messages, guardian_config)


# API-based Grounding
def grounding_binary(messages, model=DEFAULT_MODEL):
    """Perform binary grounding evaluation using an external API."""
    logger.info("Sending grounding request to external API.")
    response = requests.post(
        GRANITE_API_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": 1024 * 8, "temperature": 0, "seed": 42},
        },
        stream=False,
    )
    response.raise_for_status()
    logger.info("Received response from API.")

    output = ""
    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            logger.error(f"Error in response: {body['error']}")
            raise Exception(body["error"])
        if not body.get("done", False):
            output += body.get("message", {}).get("content", "")
        else:
            logger.info("Grounding binary evaluation complete.")
            return {"content": output}


def grounding_binary_main(context_text, response_text):
    """Wrapper for binary grounding evaluation."""
    logger.info("Preparing messages for grounding binary evaluation.")
    messages = get_message_gg_ootb(context_text, response_text, "groundedness")
    result = grounding_binary(messages)
    logger.info("Binary grounding evaluation completed successfully.")
    return result["content"]


if __name__ == "__main__":
    # Example usage
    context_text = """Eat (1964) is a 45-minute underground film created by Andy Warhol 
    and featuring painter Robert Indiana, filmed on Sunday, February 2, 1964, in Indiana's studio. 
    The film was first shown by Jonas Mekas on July 16, 1964, at the Washington Square Gallery at 
    530 West Broadway. Jonas Mekas (December 24, 1922 – January 23, 2019) was a Lithuanian-American 
    filmmaker, poet, and artist who has been called "the godfather of American avant-garde cinema". 
    Mekas's work has been exhibited in museums and at festivals worldwide."""
    response_text = "Eat (1964) a 45-minute underground film created by Andy Warhol"

    try:
        # Binary grounding evaluation
        logger.info("Starting binary grounding evaluation.")
        is_risk = grounding_binary_main(context_text, response_text)
        logger.info(f"Risk detected: {is_risk}")

        # Probabilistic grounding evaluation
        # Uncomment the next two lines if needed
        # logger.info("Starting probabilistic grounding evaluation.")
        # is_risk, prob_of_risk = grounding_with_probability(context_text, response_text)
        # logger.info(f"Risk detected: {is_risk}, Probability of risk: {prob_of_risk:.3f}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")