import argparse
from glob import glob
import json
import multiprocessing
import os
import os.path as osp
from prompt import EDIT_CATEGORIES, SEQUENCE_TEMPLATE, COMPOUND_TEMPLATE, SIMPLIFY_TEMPLATE
from tqdm import tqdm
from loguru import logger

from enum import Enum
from pydantic import BaseModel
from typing import Literal

from complex_edit.utils import (
    CLIENT_OPENAI,
    completion_retry,
    compute_usage,
    dict_sum,
    encode_msgs,
    retry_instant_decorator,
    setup_logger,
)


cat_names = [cat_name for _, cat in EDIT_CATEGORIES for cat_name, _ in cat]
OperationName = Enum('OperationName', {cat_name: cat_name for cat_name in cat_names})
STEP_NUM = None


class SimpleOperation(BaseModel):
    name: OperationName
    instruction: str


class Sequence(BaseModel):
    sequence: list[SimpleOperation]


class CoTSequence(BaseModel):
    reasoning: str
    sequence: list[SimpleOperation]


class CoTCompound(BaseModel):
    reasoning: str
    compound_instruction: str


class Simplied(BaseModel):
    need_simplication: Literal["Yes", "No"]
    instruction: str


def _check_simpleOperation(dct):
    return dct["instruction"][0].isalpha() and dct["instruction"][-1] == "."


def _check_sequence(dct):
    for step in dct["sequence"]:
        if not _check_simpleOperation(step):
            return False

    return True


def _check_cotsequence(dct):
    for step in dct["sequence"]:
        if not _check_simpleOperation(step):
            return False

    return dct["reasoning"][0].isalpha() and dct["reasoning"][-1] == "."


def _check_cotcompound(dct):
    return dct["reasoning"][0].isalpha() and \
        dct["reasoning"][-1] == "." and \
        dct["compound_instruction"][0].isalpha() and \
        dct["compound_instruction"][-1] == "."


def _check_simplied(dct):
    return dct["instruction"][0].isalpha() and dct["instruction"][-1] == "."


FORMAT2CHECKER = {
    SimpleOperation: _check_simpleOperation,
    Sequence: _check_sequence,
    CoTSequence: _check_cotsequence,
    CoTCompound: _check_cotcompound,
    Simplied: _check_simplied
}


EXAMPLE_IMAGE_PATH = "./imgs/example_image.png"
EXAMPLE_CoT_SEQUENCE = {
    "reasoning": "The image depicts a whimsical scene with a girl jumping in a field of sunflowers, surrounded by \
butterflies under a clear sky. To create a more complex and dynamic scene, we introduce various elements and \
effects in a logical sequence. First, we replace the sunflower field with a grassy field to set a neutral \
background. Adding a full moon establishes a shift from a daytime setting to a nighttime atmosphere. Changing the \
moon's color to red enhances the dramatic effect. Removing the butterflies ensures consistency in the transformed \
scene. The girl's dress is altered to a gothic black dress to match the new theme, and her facial expression \
is changed to a grim look to further reinforce the dark aesthetic. Finally, applying a twilight filter blends \
the modifications together, ensuring a cohesive transformation. Each step logically builds upon the previous one, \
maintaining consistency and avoiding loss of critical information.",
    "sequence": [
        {"name": "Change Background", "instruction": "Replace the sunflower field with a grassy field."},
        {"name": "Add an Object", "instruction": "Add a bright full moon in the sky."},
        {"name": "Change Color", "instruction": "Change the color of the moon to a deep red."},
        {"name": "Remove an Object", "instruction": "Remove the butterflies."},
        {"name": "Replace an Object", "instruction": "Change the girl's dress to a gothic black dress."},
        {"name": "Change Facial Expression", "instruction": "Make the girl's face expression grim."},
        {"name": "Apply Filter/Weather", "instruction": "Apply a twilight filter."},
        {"name": "Add Text", "instruction": "Add a text saying 'Eerie Twilight'."},
    ]
}

# Type checking
CoTSequence.model_validate(EXAMPLE_CoT_SEQUENCE)


EXAMPLE_CoT_COMPOUND = {
    "reasoning": "The sequence of instructions transforms a bright sunflower field into a darker, eerie twilight \
setting. The sunflower field is replaced with a grassy field, shifting the environment to a more neutral, open \
landscape. A full moon is added to the sky and then changed to a deep red, reinforcing a mysterious atmosphere. \
The removal of butterflies eliminates a lively element, further supporting the tone shift. The girl's dress is \
replaced with a gothic black dress, and her expression is altered to a grim look, aligning her appearance with the \
overall dark aesthetic. Add a text saying 'Eerie Twilight'. Finally, a twilight filter is applied, adjusting the \
lighting and color tones to unify the transformed scene.",
    "compound_instruction": "Change the sunflower field background to a grassy field. Add a deep red full moon in the \
sky. Remove the butterflies. Change the girl's dress to a gothic black dress and make her expression grim. Apply a \
twilight filter and enhance the eerie atmosphere."
}


# Type checking
CoTCompound.model_validate(EXAMPLE_CoT_COMPOUND)


EXAMPLE_SIMPLIFIED_POS_INPUT = "Apply a sunny filter onto the image to enhance overall skies and lighting."
EXAMPLE_SIMPLIFIED_POS_OUTPUT = {
    "need_simplication": "Yes",
    "instruction": "Apply a sunny filter."
}
EXAMPLE_SIMPLIFIED_NEG_INPUT = "Alter the palm tree leaves to a more vibrant green."
EXAMPLE_SIMPLIFIED_NEG_OUTPUT = {
    "need_simplication": "No",
    "instruction": "Alter the palm tree leaves to a more vibrant green."
}

# Type checking
Simplied.model_validate(EXAMPLE_SIMPLIFIED_POS_OUTPUT)
Simplied.model_validate(EXAMPLE_SIMPLIFIED_NEG_OUTPUT)


def build_option_prompt():
    prompt = ""

    for idx, (meta_cat, cats) in enumerate(EDIT_CATEGORIES):
        prompt += f"{idx + 1}. {meta_cat}\n"
        for cat_name, cat_desc in cats:
            prompt += f"    * {cat_name}: {cat_desc}\n"

    return prompt


def build_msgs_sequence(image_path, if_example=True):
    assert STEP_NUM is not None

    msgs = [
        {
            "role": "system",
            "content": [("text", SEQUENCE_TEMPLATE.format(num=STEP_NUM, options=build_option_prompt()))]
        },
        {
            "role": "user",
            "content": [("image", image_path)]
        }
    ]

    if if_example:
        example_output_str = json.dumps(EXAMPLE_CoT_SEQUENCE)

        msgs = msgs[:1] + [
            {
                "role": "user",
                "content": [("image", EXAMPLE_IMAGE_PATH)]
            },
            {
                "role": "assistant",
                "content": [("text", example_output_str)]
            }
        ] + msgs[1:]

    return encode_msgs(msgs)


def build_msgs_simplify(inst, if_example=True):
    assert isinstance(inst, str), inst

    msgs = [
        {
            "role": "system",
            "content": [("text", SIMPLIFY_TEMPLATE)]
        },
        {
            "role": "user",
            "content": [("text", inst)]
        }
    ]

    if if_example:
        msgs = msgs[:1] + [
            {
                "role": "user",
                "content": [("text", EXAMPLE_SIMPLIFIED_POS_INPUT)]
            },
            {
                "role": "assistant",
                "content": [("text", json.dumps(EXAMPLE_SIMPLIFIED_POS_OUTPUT))]
            },
            {
                "role": "user",
                "content": [("text", EXAMPLE_SIMPLIFIED_NEG_INPUT)]
            },
            {
                "role": "assistant",
                "content": [("text", json.dumps(EXAMPLE_SIMPLIFIED_NEG_OUTPUT))]
            },
        ] + msgs[1:]

    return encode_msgs(msgs)


def build_msgs_compound(image_path, sequence, if_example=True):
    assert isinstance(sequence, list)

    msgs = [
        {
            "role": "system",
            "content": [("text", COMPOUND_TEMPLATE)]
        },
        {
            "role": "user",
            "content": [
                ("image", image_path),
                ("text", json.dumps(sequence))
            ]
        }
    ]

    if if_example:
        example_output_str = json.dumps(EXAMPLE_CoT_COMPOUND)

        msgs = msgs[:1] + [
            {
                "role": "user",
                "content": [
                    ("image", EXAMPLE_IMAGE_PATH),
                    ("text", json.dumps(EXAMPLE_CoT_SEQUENCE["sequence"]))
                ]
            },
            {
                "role": "assistant",
                "content": [("text", example_output_str)]
            }
        ] + msgs[1:]

    return encode_msgs(msgs)


@retry_instant_decorator
def process_one_image(args):
    image_path, save_path = args

    msgs_sequence = build_msgs_sequence(image_path, STEP_NUM)

    resp = completion_retry(
        client=CLIENT_OPENAI,
        model_name="gpt-4o-2024-11-20",
        msgs=msgs_sequence,
        max_completion_tokens=1024,
        temperature=1.15,
        response_format=CoTSequence,
    )
    result = json.loads(resp.choices[0].message.content)
    assert len(result["sequence"]) == STEP_NUM, (image_path, result["sequence"])
    token_usage, cost_usage = compute_usage(resp)

    result["original_sequence"] = result.pop("sequence")
    result["sequence"] = []
    for step in result["original_sequence"]:
        name, inst = step["name"], step["instruction"]
        msgs_simplify = build_msgs_simplify(inst)
        resp = completion_retry(
            client=CLIENT_OPENAI,
            model_name="gpt-4o-2024-11-20",
            msgs=msgs_simplify,
            max_completion_tokens=256,
            response_format=Simplied,
        )

        simplified = resp.choices[0].message.parsed.model_dump()
        simplified_inst = simplified["instruction"] if simplified["need_simplication"] == "Yes" else inst
        result["sequence"].append({"name": name, "instruction": simplified_inst})

        token_usage2, cost_usage2 = compute_usage(resp)
        token_usage = dict_sum([token_usage, token_usage2])
        cost_usage = dict_sum([cost_usage, cost_usage2])

    result["compound"] = [
        {
            "reasoning": "none",
            "compound_instruction": result["sequence"][0]["instruction"]
        }
    ]

    for i in range(1, len(result["sequence"])):
        seq = result["sequence"][: i + 1]
        msgs_compound = build_msgs_compound(image_path, seq)
        resp = completion_retry(
            client=CLIENT_OPENAI,
            model_name="gpt-4o-2024-11-20",
            msgs=msgs_compound,
            max_completion_tokens=256,
            response_format=CoTCompound,
        )
        compound = resp.choices[0].message.parsed.model_dump()
        result["compound"].append(compound)

        token_usage2, cost_usage2 = compute_usage(resp)
        token_usage = dict_sum([token_usage, token_usage2])
        cost_usage = dict_sum([cost_usage, cost_usage2])

    json.dump(result, open(save_path, "w"), indent=4)
    return token_usage, cost_usage


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image editing model.")
    parser.add_argument("--path", "-p", type=str, required=True,
                        help="Path to the directory containing the input images.")
    parser.add_argument("--output-path", "-o", type=str, required=True, help="Path to save the generated instructions.")
    parser.add_argument("--max-complexity", "-c", type=int, default=8, help="Max complexity level")
    parser.add_argument("--num-processes", type=int, default=16, help="Number of processes.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    STEP_NUM = args.max_complexity

    os.makedirs(args.output_path, exist_ok=True)

    setup_logger(output=osp.join(args.output_path, "log.txt"))

    image_paths = sorted(glob(osp.join(args.path, "*.png")))
    save_paths = [
        osp.join(args.output_path, f"{image_path.split('/')[-1].split('.')[0]}.json")
        for image_path in image_paths
    ]

    with multiprocessing.Pool(processes=16) as pool:
        with tqdm(total=len(image_paths), desc="Generating instructions") as progress:
            total_token_usage = {"input": 0, "reasoning": 0, "output": 0}
            total_cost_usage = {"input": 0, "reasoning": 0, "output": 0, "total": 0}

            for idx, (token_usage, cost_usage) in enumerate(pool.imap_unordered(
                process_one_image,
                zip(image_paths, save_paths)
            )):
                progress.update(1)

                total_token_usage = dict_sum([total_token_usage, token_usage])
                total_cost_usage = dict_sum([total_cost_usage, cost_usage])

    avg_token_usage = {k: v / len(image_paths) for k, v in total_token_usage.items()}
    avg_cost_usage = {k: v / len(image_paths) for k, v in total_cost_usage.items()}

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
