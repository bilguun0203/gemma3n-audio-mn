# Gemma 3n Audio Fine-tuning for Mongolian

Fine-tuning Google's Gemma 3n multimodal model for audio transcription and translation between English and Mongolian languages.

## Features

- **Audio Transcription**: Transcribe English and Mongolian speech to text
- **Cross-language Translation**: Translate audio from English to Mongolian and vice versa
- **Multi-task Training**: Supports various instruction formats for flexible audio processing
- **LoRA Fine-tuning**: Efficient parameter-efficient training using LoRA adapters

## Datasets

The model is trained on multiple Mongolian and English audio datasets:

- **MBSpeech**: Mongolian speech dataset ([`bilguun/mbspeech`](https://huggingface.co/datasets/bilguun/mbspeech))
- **TED Talks**: English-Mongolian parallel audio ([`bilguun/ted_talks_en_mn_split`](https://huggingface.co/datasets/bilguun/ted_talks_en_mn_split))
- **Common Voice**: Mozilla Common Voice Mongolian dataset ([`mozilla-foundation/common_voice_17_0`](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) - `mn` subset)

## Demo

Fine-tuned model can be found at: [`bilguun/gemma-3n-E2B-it-audio-en-mn`](https://huggingface.co/bilguun/gemma-3n-E2B-it-audio-en-mn)

Run the fine-tuned model in a Jupyter Notebook:

```bash
jupyter notebook gemma_3n_audio_demo.ipynb
```

## Fine-tuning

To fine-tune the model:

```bash
python fine_tune_gemma3n_on_audio.py
```

## Sample Audio

The `audio_samples/` directory contains example audio files in both English (`en1.wav`, `en2.wav`, `en3.wav`) and Mongolian (`mn1.wav`, `mn2.wav`, `mn3.wav`) for testing.

## Requirements

- Python 3.8+
- PyTorch < 2.7
- Transformers < 4.55
- CUDA-capable GPU (recommended)
