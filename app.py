import gc

import gradio as gr
import spaces
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


BASE_GEMMA_MODEL_ID = "google/gemma-3n-E2B-it"
GEMMA_MODEL_ID = "bilguun/gemma-3n-E2B-it-audio-en-mn"

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(BASE_GEMMA_MODEL_ID, device_map="cuda")
model = AutoModelForImageTextToText.from_pretrained(GEMMA_MODEL_ID, device_map="cuda")

if hasattr(model, "eval"):
    model.eval()

print("Model loaded successfully!")


@spaces.GPU
def process_audio(audio_file, prompt_type):
    if audio_file is None:
        return "Please upload an audio file."

    prompts = {
        "Transcribe": "Transcribe this audio.",
        "Transcribe EN to MN": "Transcribe this audio into English and translate into Mongolian.",
        "Transcribe MN to EN": "Transcribe this audio into Mongolian and translate into English.",
    }

    selected_prompt = prompts[prompt_type]

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_file},
                    {"type": "text", "text": selected_prompt},
                ],
            }
        ]

        with torch.no_grad():
            input_ids = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            input_ids = {
                k: v.to(model.device, dtype=torch.long if "input_ids" in k else v.dtype)
                for k, v in input_ids.items()
            }

            outputs = model.generate(
                **input_ids,
                max_new_tokens=128,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

            input_length = input_ids["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]

            text = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del input_ids, outputs, generated_tokens
        gc.collect()

        return text[0]

    except Exception as e:
        return f"Error processing audio: {str(e)}"


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
            )

    process_btn.click(
        fn=process_audio,
        inputs=[audio_input, prompt_dropdown],
        outputs=output_text,
    )

    gr.Examples(
        examples=[
            ["./audio_samples/en1.wav", "Transcribe"],
            ["./audio_samples/en3.wav", "Transcribe EN to MN"],
            ["./audio_samples/mn2.wav", "Transcribe"],
            ["./audio_samples/mn2.wav", "Transcribe MN to EN"],
        ],
        inputs=[audio_input, prompt_dropdown],
        outputs=output_text,
        fn=process_audio,
        cache_examples=True,
        cache_mode="eager",  # Cache examples eagerly for model warmup
        label="Example Audio Files",
    )

if __name__ == "__main__":
    demo.launch()
