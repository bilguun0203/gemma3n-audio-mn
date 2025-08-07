from threading import Thread

import gradio as gr
import spaces
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation.streamers import TextIteratorStreamer

BASE_GEMMA_MODEL_ID = "google/gemma-3n-E2B-it"
GEMMA_MODEL_ID = "bilguun/gemma-3n-E2B-it-audio-en-mn"

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(BASE_GEMMA_MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

print("Model loaded successfully!")


@spaces.GPU(duration=60)
@torch.inference_mode()
def process_audio(audio_file, prompt_type, max_tokens):
    if audio_file is None:
        return "Please upload an audio file."

    prompts = {
        "Transcribe": "Transcribe this audio.",
        "Transcribe EN to MN": "Transcribe this audio into English and translate into Mongolian.",
        "Transcribe MN to EN": "Transcribe this audio into Mongolian and translate into English.",
    }

    selected_prompt = prompts[prompt_type]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_file},
                {"type": "text", "text": selected_prompt},
            ],
        }
    ]

    input_ids = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = input_ids.to(model.device, dtype=model.dtype)

    streamer = TextIteratorStreamer(
        processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        disable_compile=True,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    output = ""
    for delta in streamer:
        output += delta
        yield output


with gr.Blocks(title="Gemma 3n Audio Transcription & Translation") as demo:
    gr.Markdown("# Gemma 3n E2B - English-Mongolian Audio Transcription & Translation")
    gr.Markdown(
        "Upload an audio file and select the processing type to get transcription and/or translation."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Audio",
                type="filepath",
                sources=["upload", "microphone"],
                max_length=300,
            )

            prompt_dropdown = gr.Dropdown(
                choices=["Transcribe", "Transcribe EN to MN", "Transcribe MN to EN"],
                value="Transcribe",
                label="Prompt Type",
            )

            process_btn = gr.Button("Process Audio", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Output",
                lines=10,
                max_lines=20,
                placeholder="Transcribed text will appear here...",
                show_copy_button=True,
                interactive=False,
            )

    with gr.Row():
        with gr.Accordion("Additional Settings", open=False):
            max_tokens_slider = gr.Slider(
                minimum=16,
                maximum=512,
                value=128,
                step=16,
                label="Max New Tokens",
                info="Maximum number of tokens to generate",
            )

    process_btn.click(
        fn=process_audio,
        inputs=[audio_input, prompt_dropdown, max_tokens_slider],
        outputs=output_text,
    )

    gr.Examples(
        examples=[
            [
                "https://github.com/bilguun0203/gemma3n-audio-mn/raw/refs/heads/main/audio_samples/en1.wav",
                "Transcribe",
                128,
            ],
            [
                "https://github.com/bilguun0203/gemma3n-audio-mn/raw/refs/heads/main/audio_samples/en3.wav",
                "Transcribe EN to MN",
                128,
            ],
            [
                "https://github.com/bilguun0203/gemma3n-audio-mn/raw/refs/heads/main/audio_samples/mn2.wav",
                "Transcribe",
                128,
            ],
            [
                "https://github.com/bilguun0203/gemma3n-audio-mn/raw/refs/heads/main/audio_samples/mn2.wav",
                "Transcribe MN to EN",
                128,
            ],
        ],
        inputs=[
            audio_input,
            prompt_dropdown,
            max_tokens_slider,
        ],
        outputs=output_text,
        fn=process_audio,
        cache_examples=True,
        cache_mode="eager",  # Cache examples eagerly for model warmup
        label="Example Audio Files",
    )

if __name__ == "__main__":
    demo.launch()
