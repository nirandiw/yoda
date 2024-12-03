from prompts import guardrail_system_message, user_input
from utils import GPT_MODEL

import pandas as pd
import json
import logging
from auth import get_openai_client

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def guardrail_groundedness(context, query, genai_response,client):
    logger.info("Processing guardrail groundedness")
    user_input_filled = user_input.format(
        context=context,
        query=query,
        message=genai_response
    )
    
    messages = [
        { "role": "system", "content": guardrail_system_message},
        { "role": "user", "content": user_input_filled}
    ]

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.7,
        n=10
    )
    logger.info("Guardrail groundedness processing completed")
    return response.choices

def is_sentence_grounded(item):
    score_sum = sum(
            item.get(key, 0)
            for key in ["factualAccuracy", "relevance", "policyCompliance", "contextualCoherence"]
        )
    groundedness = score_sum == 4
    return groundedness

def is_answer_grounded(response_data):
    """Check if the answer is grounded based on the response data."""
    return all(is_sentence_grounded(item) for item in response_data)

def main():

    client = get_openai_client()
    
    context_text = """Eat (1964) is a 45-minute underground film created by Andy Warhol and featuring painter Robert Indiana, filmed on Sunday, February 2, 1964, in Indiana's studio. The film was first shown by Jonas Mekas on July 16, 1964, at the Washington Square Gallery at 530 West Broadway.
    Jonas Mekas (December 24, 1922 – January 23, 2019) was a Lithuanian-American filmmaker, poet, and artist who has been called "the godfather of American avant-garde cinema". Mekas's work has been exhibited in museums and at festivals worldwide."""
    response_text = "Eat means putting food to your tummy"
    query = "What is Eat?"
    query = "What types of cover are available under the pet insurance plans?",
    context_text = "The following plans are offered in this booklet:\nPlan name Type of cover\nClassic Cover\u2022  cover  for specified accidental injuries  and illnesses\n\u2022  up to $15,000 annual benefit limit  per policy period  \n\u2022  sub-limits  on consultations , tick \nparalysis, hip joint surgery , emergency boarding \nand cruciate ligament conditions\n\u2022  up to 70% and 80% benefit percentage  options \n(depending on selection)\n\u2022  optional non-insurance benefit for routine care  \n\u2022 optional cover  for booster care \nPlus Cover\u2022  cover  for specified accidental injuries  and illnesses\n\u2022  up to $30,000 annual benefit limit  per policy period\n\u2022 no sub-limits\n\u2022  up to 80% or 90% benefit percentage  options \n(depending on selection)\n\u2022  optional non-insurance benefit for routine care  \n\u2022 optional cover  for booster care",
    response_text= "There are two types of cover available: Classic Cover and Plus Cover. Classic Cover provides cover for specified accidental injuries and illnesses with an annual benefit limit of up to $15,000, includes sub-limits, and offers benefit percentage options of 70% or 80%. Plus Cover also covers specified accidental injuries and illnesses but with an annual benefit limit of up to $30,000, has no sub-limits, and offers benefit percentage options of 80% or 90%. Both plans offer optional non-insurance benefits for routine care and optional cover for booster care.",
  
    
    choices = guardrail_groundedness(context_text,query, response_text, client )
    logger.info(f"Grounding information {choices[0].message.content.strip('`json')}")
    response_json = choices[0].message.content.strip('`json')
    response_data = json.loads(response_json)
    for item in response_data:
        groundedness = is_sentence_grounded(item)
        logger.info(f" Sentence: {item['sentence']}... Grounded??: {groundedness}")



if __name__ == "__main__":

    main()