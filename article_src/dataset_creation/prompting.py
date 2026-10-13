"""
Taken from source of article BLESS (https://arxiv.org/pdf/2310.15773.pdf) https://github.com/ZurichNLP/BLESS. under
MIT License.
"""

import json
from typing import List, Dict, Optional

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


def load_predefined_prompt(prompt_json_path) -> Dict:
    """
    Loads a predefined prompt from a JSON file when the `--prompt_json` argument is provided.
    Relevant arguments are then updated to reflect the loaded prompt.
    """
    with open(prompt_json_path, "r") as f:
        prompt_args = json.load(f)
    return prompt_args


def prepare_prompted_inputs(
    inputs: List[str],
    prompt_prefix: str = "Simplify the following sentence:",
    prompt_suffix: str = r"Complex: {input}\nSimple:",
    examples: Optional[List[Dict]] = None,
    example_selector: Optional[BaseExampleSelector] = None,
    example_prompt: PromptTemplate = None,
    example_separator: str = r"\n\n",
    prompt_format: str = "prefix_initial",
) -> List[str]:
    """
    Constructs few-shot prompts for a batch of inputs.

    Args:
        inputs :: a list of strings to be prompted
        prompt_prefix :: a string to be added to the beginning of the prompt
        prompt_suffix :: a string to be added to the end of the prompt. Should contain the input variable name.
        examples :: a list of dictionaries containing the examples to be used in the prompt
        example_selector :: an ExampleSelector object to select examples from the examples list
        example_prompt :: a PromptTemplate object specifying the format of the examples in the prompt.
        example_separator :: a string to be added between each example in the prompt
        prompt_format :: a string specifying the format of the prompt.
            Options are prefix_initial or prefix_every:
            - if prefix_initial, the prefix is added to the beginning of the prompt and the examples are separated by
            example_separator.
            - if prefix_every, the example_separator is modified to include the prefix at the beginning of each example.
    """

    prompted_inputs = []

    if examples is None and example_selector is None:
        # Zero-shot setting
        for i in inputs:
            prompt = PromptTemplate.from_template(
                "\n".join([prompt_prefix, prompt_suffix])
            )
            zsp = prompt.format(input=i)

            prompted_inputs.append(zsp)
    else:
        # Few-shot setting
        for i in inputs:
            if prompt_format == "prefix_initial":
                few_shot_prompt = FewShotPromptTemplate(
                    examples=None if example_selector is None else examples,
                    example_selector=example_selector,  # use an ExampleSelector instead of examples.
                    example_prompt=example_prompt,  # examples format
                    prefix=prompt_prefix,
                    suffix=prompt_suffix,
                    input_variables=[
                        "input"
                    ],  # the variables that the overall prompt expects
                    example_separator=example_separator,
                    # string used to join the prefix, examples, and suffix together
                )
            elif (
                prompt_format == "prefix_every"
            ):  # warning: this is a hacky way to add the prefix to every example
                # If using prefix_every, the examples separator is expanded to include the prefix.
                few_shot_prompt = FewShotPromptTemplate(
                    examples=None if example_selector is None else examples,
                    example_selector=example_selector,  # use an ExampleSelector instead of examples.
                    example_prompt=example_prompt,  # examples format
                    prefix=prompt_prefix,  # prefix is added to the example_separator instead
                    suffix=prompt_suffix,
                    input_variables=[
                        "input"
                    ],  # the variables that the overall prompt expects
                    example_separator=example_separator
                    + prompt_prefix
                    + example_separator,
                    # string used to join the prefix, examples, and suffix together
                )
            else:
                raise ValueError(f"Argument {prompt_format} not recognized.")

            # fill in the prompt template with the input
            fsp = few_shot_prompt.format(input=i)
            # if using `--prompt_format=prefix_every`, the prefix prompt is repeated twice at the start,
            # so we replace it with a single instance
            fsp = fsp.replace(
                prompt_prefix + example_separator + prompt_prefix, prompt_prefix
            ).strip()
            prompted_inputs.append(fsp)

    return prompted_inputs
