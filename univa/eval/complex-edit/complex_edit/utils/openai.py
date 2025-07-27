from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI, NOT_GIVEN
from loguru import logger
from tenacity import (
    retry,
    wait_random_exponential,
)  # for exponential backoff


CLIENT_OPENAI = OpenAI(
    api_key="<OPENAI_API_KEY>"
)


def compute_usage(response):
    usage = response.usage.to_dict()
    input = usage["prompt_tokens"]
    reasoning = usage.get("completion_tokens_details", {"reasoning_tokens": 0})["reasoning_tokens"]
    output = usage["completion_tokens"] - reasoning

    cost = {
        "input": input * 5 / 10 ** 6,
        "reasoning": reasoning * 0 / 10 ** 6,
        "output": output * 20 / 10 ** 6,
    }

    cost["total"] = sum(cost.values())

    return {"input": input, "reasoning": reasoning, "output": output}, cost


def encode_image(image):
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG", quality=100)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def encode_msgs(msgs):
    encoded_msgs = []

    for msg in msgs:
        encoded_msg = {"role": msg["role"], "content": []}

        for single_content in msg["content"]:
            content_type, content_body = single_content
            assert content_type in ["text", "image"]
            if content_type == "text":
                encoded_msg["content"].append(
                    {"type": "text", "text": content_body}
                )
            elif content_type == "image":
                encoded_msg["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(content_body)}",
                            "detail": "high"
                        }
                    }
                )
            else:
                raise ValueError(f"Unrecognized content_type: {content_type}")

        encoded_msgs.append(encoded_msg)

    return encoded_msgs


def _log_when_fail(retry_state):
    logger.debug(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )


retry_decorator = retry(
    wait=wait_random_exponential(min=1, max=60),
    before_sleep=_log_when_fail
)

# another decorator. but no wait time. infinite instant retry
retry_instant_decorator = retry(
    before_sleep=_log_when_fail
)


@retry_decorator
def completion_retry(
    client, model_name, msgs,
    max_completion_tokens=512,
    temperature=1.0,
    n=1,
    response_format=NOT_GIVEN
):
    resp = client.beta.chat.completions.parse(
        model=model_name,
        messages=msgs,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        n=n,
        response_format=response_format
    )

    if resp.choices[0].finish_reason != "stop":
        raise ValueError(
            "Generation finish reason: {}. {}".format(
                resp.choices[0].finish_reason,
                resp.usage.to_dict()
            )
        )

    return resp
