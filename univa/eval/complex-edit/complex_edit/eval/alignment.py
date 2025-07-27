from copy import deepcopy
from functools import partial
import json
from loguru import logger
import multiprocessing
import os.path as osp
from tqdm import tqdm

from pydantic import BaseModel
from typing import Literal

from complex_edit.utils import (
    CLIENT_OPENAI,
    completion_retry,
    compute_usage,
    dict_mean,
    dict_sum,
    encode_msgs,
    retry_instant_decorator,
)
from .prompt_alignment import (
    ALIGNMENT_PROMPT_WO_RUBRIC,
    ALIGNMENT_PROMPT_W_RUBRIC,
    PROMPT_TEMPLATE
)


class Alignment(BaseModel):
    instruction_following: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    identity_preservation: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


class CoTAlignment(BaseModel):
    reasoning: str
    instruction_following: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    identity_preservation: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def build_msgs(input_image, output_image, instruction, system_prompt, prompt_template):
    """
    Constructs and encodes a list of messages for GPT.

    Args:
        input_image (Union[PIL.Image.Image, str]): The input image, either as a PIL.Image object
            or a file path to the image.
        output_image (Union[PIL.Image.Image, str]): The output image, either as a PIL.Image object
            or a file path to the image.
        instruction (str): The edit instruction.
        system_prompt (str): The system's prompt text to guide the conversation.
        prompt_template (str): A template string for formatting the instruction into the user message.

    Returns:
        Encoded messages for GPT.
    """
    msgs = [
        {
            "role": "system",
            "content": [("text", system_prompt)]
        },
        {
            "role": "user",
            "content": [
                ("image", input_image),
                ("image", output_image),
                ("text", prompt_template.format(instruction=instruction))
            ]
        }
    ]

    return encode_msgs(msgs)


@retry_instant_decorator
def eval_one_alignment(
    args,
    response_format, if_resume,
    n, m,
    system_prompt, prompt_template
):
    """
    Evaluates the alignment (e.g. instruction_following, identity_preservation).

    Args:
        args (tuple): A tuple containing the following elements:
            - input_image (Union[PIL.Image.Image, str]): The input image, either as a PIL.Image object
                or a file path to the image.
            - output_image (Union[PIL.Image.Image, str]): The output image, either as a PIL.Image object
                or a file path to the image.
            - instruction (str): The edit instruction.
            - save_path (str): Path to save the evaluation results.
        response_format (str): The format of the response expected from the model.
        if_resume (bool): Whether to resume from a previously saved result if it exists.
        n (int): Total number of measurements for one sample.
        m (int): Maximum number of responses per call. e.g. n = 10 and m = 5, then 2 calls will be made.
        system_prompt (str): The system prompt to use for the model.
        prompt_template (str): The template for constructing the prompt.

    Returns:
        tuple: A tuple containing:
            - avg_result (dict): The evaluation results, including metrics "instruction_following" and
                "identity_preservation".
            - token_usage (dict): A dictionary summarizing token usage statistics.
            - cost_usage (dict): A dictionary summarizing cost usage statistics.
    """
    input_image, output_image, instruction, save_path = args

    if if_resume and osp.exists(save_path):
        avg_result = json.load(open(save_path))
        avg_result.pop("instruction", None)
        avg_result.pop("runs", None)
        return avg_result, None, None

    msgs = build_msgs(
        input_image, output_image, instruction,
        system_prompt=system_prompt,
        prompt_template=prompt_template
    )

    resps = []

    n_list = [m for _ in range(n // m)]
    if n % m != 0:
        n_list.append(n % m)

    for curr_n in n_list:
        resp = completion_retry(
            client=CLIENT_OPENAI,
            model_name="gpt-4o-2024-11-20",
            msgs=msgs,
            n=curr_n,
            response_format=response_format,
        )

        resps.append(resp)

    token_usages, cost_usages = list(zip(*[compute_usage(resp) for resp in resps]))
    token_usage = dict_sum(token_usages)
    cost_usage = dict_sum(cost_usages)

    results = []

    for choice in sum([resp.choices for resp in resps], start=[]):
        result = choice.message.parsed.model_dump()
        result["instruction_following"] = int(result["instruction_following"])
        result["identity_preservation"] = int(result["identity_preservation"])
        assert 0 <= result["instruction_following"] <= 10, result
        assert 0 <= result["identity_preservation"] <= 10, result

        results.append(result)

    assert len(results) == n, len(results)

    to_save = {
        "instruction": instruction,
        "runs": deepcopy(results),
    }

    for result in results:
        result.pop("reasoning", None)

    avg_result = dict_mean(results)
    to_save.update(avg_result)

    with open(save_path, "w") as f:
        json.dump(to_save, f, indent="    ")

    return avg_result, token_usage, cost_usage


def get_system_prompt(if_rubric, if_cot):
    system_prompt = ALIGNMENT_PROMPT_W_RUBRIC if if_rubric else ALIGNMENT_PROMPT_WO_RUBRIC
    if if_cot:
        system_prompt += "\nExplain your reasoning before answering the questions."

    return system_prompt


def get_result_folder_name(if_rubric, if_cot):
    folder_name = "alignment"
    if if_rubric:
        folder_name += "_rubric"
    if if_cot:
        folder_name += "_cot"

    return folder_name


class AlignmentEvaluator:
    def __init__(self, if_rubric=False, if_cot=False, if_resume=False, n=20, m=5, num_processes=4):
        """
        Args:
            if_rubric (bool, optional): Flag to indicate if rubric for evaluation is used. Defaults to False.
            if_cot (bool, optional): Flag to indicate if CoT is used. Defaults to False.
            if_resume (bool, optional): Flag to indicate if the process should resume from a previous state.
                Defaults to False.
            n (int, optional): Total number of measurements for one sample. Defaults to 20.
            m (int, optional): Maximum number of responses per call. e.g. n = 10 and m = 5, then 2 calls will be made.
                Defaults to 5.
            num_processes (int, optional): Number of parallel processes to use. Defaults to 4.
        """
        self.if_rubric = if_rubric
        self.if_cot = if_cot
        self.if_resume = if_resume
        self.n = n
        self.m = m
        self.num_processes = num_processes

        self.system_prompt = get_system_prompt(if_rubric, if_cot)
        self.result_folder_name = get_result_folder_name(if_rubric, if_cot)

        self.response_format = CoTAlignment if self.if_cot else Alignment

    def eval(self, input_images, output_images, instructions, save_paths):
        """
        Evaluate the alignment (e.g. instruction_following, identity_preservation) between input images,
            output images, and instructions.
        Args:
            input_images (List[Union[PIL.Image.Image, str]]): A list of input images,
                either as PIL.Image objects or file paths to the images.
            output_images (List[Union[PIL.Image.Image, str]]): A list of output images,
                either as PIL.Image objects or file paths to the images.
            instructions (List[str]): A list of edit instructions corresponding to the input and output images.
            save_paths (List[str]): A list of file paths where the evaluation results will be saved.
        Returns:
            List[Any]: A list of evaluation results for each sample.
        """
        assert len(input_images) == len(output_images) == len(instructions) == len(save_paths)

        eval_func_partial = partial(
            eval_one_alignment,
            response_format=self.response_format,
            if_resume=self.if_resume,
            n=self.n, m=self.m,
            system_prompt=self.system_prompt,
            prompt_template=PROMPT_TEMPLATE
        )

        results = []

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            with tqdm(total=len(output_images), desc="Evaluating Alignment") as progress:
                total_token_usage = {"input": 0, "reasoning": 0, "output": 0}
                total_cost_usage = {"input": 0, "reasoning": 0, "output": 0, "total": 0}

                for avg_result, token_usage, cost_usage in pool.imap_unordered(
                    eval_func_partial,
                    zip(input_images, output_images, instructions, save_paths)
                ):
                    progress.update(1)
                    results.append(avg_result)

                    if token_usage is None or cost_usage is None:
                        continue

                    total_token_usage = dict_sum([total_token_usage, token_usage])
                    total_cost_usage = dict_sum([total_cost_usage, cost_usage])

            avg_token_usage = {k: v / len(output_images) for k, v in total_token_usage.items()}
            avg_cost_usage = {k: v / len(output_images) for k, v in total_cost_usage.items()}

            # keep only 5 digits after the decimal point for total and avg
            total_token_usage = {k: round(v, 5) for k, v in total_token_usage.items()}
            total_cost_usage = {k: round(v, 5) for k, v in total_cost_usage.items()}
            avg_token_usage = {k: round(v, 5) for k, v in avg_token_usage.items()}
            avg_cost_usage = {k: round(v, 5) for k, v in avg_cost_usage.items()}

            logger.info(
                f"Average Cost usage: {avg_cost_usage}, Total Cost usage: {total_cost_usage}"
            )
            logger.info(
                f"Average Token usage: {avg_token_usage}, Total Token usage: {total_token_usage}"
            )

        return results
