import argparse
import datetime
import json
import os
import time
import torch
import torchaudio

import gradio as gr
import numpy as np
import requests
import soundfile as sf

from omni_speech.conversation import default_conversation, conv_templates
from omni_speech.constants import LOGDIR
from omni_speech.utils import build_logger, server_error_msg
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder


logger = build_logger("gradio_web_server", "gradio_web_server.log")

vocoder = None

headers = {"User-Agent": "LLaMA-Omni Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, None, "", "", None)


def add_speech(state, speech, request: gr.Request):
    text = "Please directly answer the questions in the user's speech."
    text = '<speech>\n' + text
    text = (text, speech)
    state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state)


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, chunk_size, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, "", "", None)
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        template_name = "llama_3"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, "", "", None)
        return

    # Construct prompt
    prompt = state.get_prompt()

    sr, audio = state.messages[0][1][1]
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    audio = torch.tensor(audio.astype(np.float32)).unsqueeze(0)
    audio = resampler(audio).squeeze(0).numpy()
    audio /= 32768.0
    audio = audio.tolist()
    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1500),
        "stop": state.sep2,
        "audio": audio,
    }

    yield (state, "", "", None)

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        num_generated_units = 0
        wav_list = []
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    output_unit = list(map(int, data["unit"].strip().split()))
                    state.messages[-1][-1] = (output, data["unit"].strip())

                    # vocoder
                    new_units = output_unit[num_generated_units:]
                    if len(new_units) >= chunk_size:
                        num_generated_units = len(output_unit)
                        x = {"code": torch.LongTensor(new_units).view(1, -1).cuda()}
                        wav = vocoder(x, True)
                        wav_list.append(wav.detach().cpu().numpy())

                    if len(wav_list) > 0:
                        wav_full = np.concatenate(wav_list)
                        return_value = (16000, wav_full)
                    else:
                        return_value = None

                    yield (state, state.messages[-1][-1][0], state.messages[-1][-1][1], return_value)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, "", "", None)
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, "", "", None)
        return

    if num_generated_units < len(output_unit):
        new_units = output_unit[num_generated_units:]
        num_generated_units = len(output_unit)
        x = {
            "code": torch.LongTensor(new_units).view(1, -1).cuda()
        }
        wav = vocoder(x, True)
        wav_list.append(wav.detach().cpu().numpy())
    
    if len(wav_list) > 0:
        wav_full = np.concatenate(wav_list)
        return_value = (16000, wav_full)
    else:
        return_value = None

    yield (state, state.messages[-1][-1][0], state.messages[-1][-1][1], return_value)

    finish_tstamp = time.time()
    logger.info(f"{output}")
    logger.info(f"{output_unit}")


title_markdown = ("""
# 🎧 LLaMA-Omni: Seamless Speech Interaction with Large Language Models
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode, vocoder, cur_dir=None, concurrency_count=10):
    with gr.Blocks(title="LLaMA-Omni Speech Chatbot", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False)

        with gr.Row():
            audio_input_box = gr.Audio(sources=["upload", "microphone"], label="Speech Input")
            with gr.Accordion("Parameters", open=True) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, interactive=True, label="Temperature",)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max Output Tokens",)
                chunk_size = gr.Slider(minimum=10, maximum=500, value=40, step=10, interactive=True, label="Chunk Size",)

        if cur_dir is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
        gr.Examples(examples=[
            [f"{cur_dir}/examples/vicuna_1.wav"],
            [f"{cur_dir}/examples/vicuna_2.wav"],
            [f"{cur_dir}/examples/vicuna_3.wav"],
            [f"{cur_dir}/examples/vicuna_4.wav"],
            [f"{cur_dir}/examples/vicuna_5.wav"],
            [f"{cur_dir}/examples/helpful_base_1.wav"],
            [f"{cur_dir}/examples/helpful_base_2.wav"],
            [f"{cur_dir}/examples/helpful_base_3.wav"],
            [f"{cur_dir}/examples/helpful_base_4.wav"],
            [f"{cur_dir}/examples/helpful_base_5.wav"],
        ], inputs=[audio_input_box])

        with gr.Row():
            submit_btn = gr.Button(value="Send", variant="primary")
            clear_btn = gr.Button(value="Clear")

        text_output_box = gr.Textbox(label="Text Output", type="text")
        unit_output_box = gr.Textbox(label="Unit Output", type="text") 
        audio_output_box = gr.Audio(label="Speech Output")

        url_params = gr.JSON(visible=False)

        submit_btn.click(
            add_speech,
            [state, audio_input_box],
            [state]
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens, chunk_size],
            [state, text_output_box, unit_output_box, audio_output_box],
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, audio_input_box, text_output_box, unit_output_box, audio_output_box],
            queue=False
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


def build_vocoder(args):
    global vocoder
    if args.vocoder is None:
        return None
    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg).cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--vocoder", type=str)
    parser.add_argument("--vocoder-cfg", type=str)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()
    build_vocoder(args)

    logger.info(args)
    demo = build_demo(args.embed, vocoder, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )