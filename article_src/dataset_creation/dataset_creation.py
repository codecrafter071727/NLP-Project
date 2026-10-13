"""
Prompting references:
    [1] HuggingFace prompting documentation: https://huggingface.co/docs/transformers/main/tasks/prompting
    [2] BLESS prompts: https://arxiv.org/pdf/2310.15773.pdf and https://github.com/ZurichNLP/BLESS/tree/main/prompts
    [3] https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
    [4] https://huggingface.co/blog/how-to-generate
"""

import os.path

from datasets import load_dataset
from dotenv import dotenv_values
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from prompting import prepare_prompted_inputs, load_predefined_prompt

root = "..."


def generate_simplification(rows):
    batch_inputs = prepare_prompted_inputs(
        inputs=rows["text"],
        example_selector=example_selector,
        example_prompt=example_prompt,
        **predefined_prompt
    )

    # handle chat inputs differently
    human_template = "{simplified_text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=model, prompt=chat_prompt)

    batch_outputs = []
    for batch_input in batch_inputs:
        # We only get the return of the model as the text
        output = chain.invoke(batch_input, return_only_outputs=True)
        # The output text is under the "text" key
        batch_outputs.append(output["text"])

    return {"simplified_text": batch_outputs}


secrets = dotenv_values(os.path.join(root, ".env"))

api_key = secrets["OPENAI_API_KEY"]

model_name = "gpt-4-turbo-2024-04-09"

# Params based on reference [2] (e.g. identical as defaults params).
max_new_tokens = 100
temperature = 1.0
top_p = 0.9
frequency_penalty = 0.0
presence_penalty = 0.0

insurance_text_snippets = load_dataset(
    os.path.join(root, "datastore"), data_files=["insurance_text_snippets.jsonl"]
)

insurance_text_snippets_pretest = load_dataset(
    os.path.join(root, "datastore"),
    data_files=["insurance_text_snippets_pretest.jsonl"],
)

# We use Langchain wrapper to interact with OpenAI GPT models.
model = ChatOpenAI(
    openai_api_key=api_key,
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_new_tokens,
    model_kwargs={
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    },
)

predefined_prompt = load_predefined_prompt(os.path.join("prompts", "p0.json"))
# Remove three args that are use for few-shot prompting.
prompt_template = predefined_prompt.pop("prompt_template")
source_field = predefined_prompt.pop("source_field")
target_field = predefined_prompt.pop("target_field")

# Since we use the zero-shot prompt, we need to set the example selector and prompt to None.
example_selector = None
example_prompt = None

insurance_text_snippets = insurance_text_snippets.map(
    function=generate_simplification, batched=True, batch_size=10
)

insurance_text_snippets["train"].to_json(
    os.path.join(root, "datastore", "insurance_text_simplifications.jsonl"),
    force_ascii=False,
)


insurance_text_snippets_pretest = insurance_text_snippets_pretest.map(
    function=generate_simplification, batched=True, batch_size=10
)

insurance_text_snippets_pretest["train"].to_json(
    os.path.join(root, "datastore", "insurance_text_simplifications_pretest.jsonl"),
    force_ascii=False,
)
