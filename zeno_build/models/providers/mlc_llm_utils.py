"""Tools to generate from MLC-LLM."""

import re
import os
import tvm
import glob

import torch
import tqdm
import transformers

from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt
from zeno_build.prompts.prompt_utils import replace_variables
from mlc_llm.python.mlc_chat.chat_module import ChatModule


def first_idx_mismatch(str1, str2):
    """Find the first index that mismatch in two strings."""
    for i, (char1, char2) in enumerate(zip(str1, str2)):
        if char1 != char2:
            return i
    return min(len(str1), len(str2))


def generate_from_mlc_llm(
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
) -> list[str]:
    """Generate outputs from MLC-LLM.

    Args:
        full_contexts: The full contexts to generate from.
        prompt_template: The prompt template to use.
        model_config: The model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top-p value to use.
        context_length: The context length to use.

    Returns:
        The generated outputs.
    """

    print(f"begin generate_from_mlc_llm")

    chat_mod = ChatModule()
    artifact_path = "/root/dist"

    # reload the model
    model_dir = os.path.join(artifact_path, model_config.model)
    model_lib = os.path.join(model_dir, model_config.model + "-cuda.so")
    print(f"loading model from {model_lib}")
    assert os.path.exists(model_lib)

    lib = tvm.runtime.load_module(model_lib)
    assert lib is not None
    chat_mod.reload_func(lib, os.path.join(model_dir, "params"))

    # chat interface
    def get_output(stream_interval=2):
        i, cur_utf8_chars = 0, "".encode("utf-8")
        res = ""
        while not chat_mod.stopped():
            chat_mod.decode()
            if i % stream_interval == 0 or chat_mod.stopped():
                new_msg = chat_mod.get_message()
                new_utf8_chars = new_msg.encode("utf-8")
                pos = first_idx_mismatch(cur_utf8_chars, new_utf8_chars)
                print_msg = ""
                for _ in range(pos, len(cur_utf8_chars)):
                    print_msg += "\b \b"
                for j in range(pos, len(new_utf8_chars)):
                    print_msg += chr(new_utf8_chars[j])
                cur_utf8_chars = new_utf8_chars
                res += print_msg
        return res

    results = []
    for i in tqdm.trange(0, len(full_contexts)):
        context = full_contexts[i]

        if len(context.messages) == 0:
            # reset the chat
            chat_mod.reset_chat()
            user_input = ""
        else:
            user_input = context.messages[-1].content

        chat_mod.prefill(user_input)
        res = get_output()
        results.append(res)

    import pdb

    pdb.set_trace()

    print(f"finish generate_from_mlc_llm")
    return results
