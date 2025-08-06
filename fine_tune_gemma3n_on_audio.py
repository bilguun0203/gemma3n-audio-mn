import random

import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
)
from trl import (
    SFTConfig,
    SFTTrainer,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


print("\n---\nLoading datasets...\n---\n")

mbspeech = load_dataset("bilguun/mbspeech")
mbspeech["train"] = mbspeech["train"].add_column(
    "audio_language", ["mn"] * len(mbspeech["train"])
)
mbspeech = mbspeech["train"].train_test_split(test_size=0.1, seed=RANDOM_SEED)

mbspeech = mbspeech.remove_columns(["sentence_norm"])
mbspeech = mbspeech.rename_column("sentence_orig", "text")
mbspeech = mbspeech.cast_column("audio", Audio(sampling_rate=16000))

print(mbspeech)

ted_en = load_dataset("bilguun/ted_talks_en_mn_split", split="en")
ted_en = ted_en.train_test_split(test_size=0.1, seed=RANDOM_SEED)

# ted_en = ted_en.remove_columns(["subtitle_en", "subtitle_mn", "filename"])
ted_en = ted_en.remove_columns(["filename", "group"])
ted_en = ted_en.cast_column("audio", Audio(sampling_rate=16000))

print(ted_en)

ted_mn = load_dataset("bilguun/ted_talks_en_mn_split", split="mn")
ted_mn = ted_mn.train_test_split(test_size=0.1, seed=RANDOM_SEED)

# ted_mn = ted_mn.remove_columns(["subtitle_en", "subtitle_mn", "filename"])
ted_mn = ted_mn.remove_columns(["filename", "group"])
ted_mn = ted_mn.cast_column("audio", Audio(sampling_rate=16000))

print(ted_mn)

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "mn",
    split="train+validation",
    trust_remote_code=True,
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_17_0", "mn", split="test", trust_remote_code=True
)

common_voice = common_voice.remove_columns(
    [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ]
)
common_voice = common_voice.remove_columns(["variant"])
common_voice = common_voice.rename_column("sentence", "text")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

common_voice["train"] = common_voice["train"].add_column(
    "audio_language", ["mn"] * len(common_voice["train"])
)
common_voice["test"] = common_voice["test"].add_column(
    "audio_language", ["mn"] * len(common_voice["test"])
)

print(common_voice)

print("\n---\nFiltering out empty audios...\n---\n")

ted_en = ted_en.filter(lambda x: x["audio"]["array"].size > 0)
ted_mn = ted_mn.filter(lambda x: x["audio"]["array"].size > 0)


def format_mix_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    instructions = [
        {
            "instruction": "Transcribe this audio.",
            "audio_languages": ["en", "mn"],
            "output_languages": ["en", "mn"],
            "all_output_required": False,
        },
        {
            "instruction": "Transcribe this audio into Mongolian.",
            "audio_languages": ["en", "mn"],
            "output_languages": ["mn"],
            "all_output_required": False,
        },
        {
            "instruction": "Transcribe this English audio into Mongolian.",
            "audio_languages": ["en"],
            "output_languages": ["mn"],
            "all_output_required": False,
        },
        {
            "instruction": "Transcribe this audio into English.",
            "audio_languages": ["en", "mn"],
            "output_languages": ["en"],
            "all_output_required": False,
        },
        {
            "instruction": "Transcribe this Mongolian audio into English.",
            "audio_languages": ["mn"],
            "output_languages": ["en"],
            "all_output_required": False,
        },
        {
            "instruction": "Translate this audio into Mongolian.",
            "audio_languages": ["en"],
            "output_languages": ["mn"],
            "all_output_required": False,
        },
        {
            "instruction": "Translate this audio into English.",
            "audio_languages": ["mn"],
            "output_languages": ["en"],
            "all_output_required": False,
        },
        {
            "instruction": "Transcribe this audio into English, and then translate it into Mongolian.",
            "audio_languages": ["en"],
            "output_languages": ["en", "mn"],
            "all_output_required": True,
        },
        {
            "instruction": "Transcribe this audio into Mongolian, and then translate it into English.",
            "audio_languages": ["mn"],
            "output_languages": ["mn", "en"],
            "all_output_required": True,
        },
    ]
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"]
        audio_language = samples["audio_language"][idx]

        text_en = None
        text_mn = None
        if "text_en" in samples and "text_mn" in samples:
            text_en = str(samples["text_en"][idx])
            text_mn = str(samples["text_mn"][idx])
        elif audio_language == "en":
            text_en = str(samples["text"][idx])
        elif audio_language == "mn":
            text_mn = str(samples["text"][idx])

        en_to_both = text_mn and text_en and audio_language == "en"
        mn_to_both = text_mn and text_en and audio_language == "mn"
        en_to_en = (
            (not en_to_both and not mn_to_both)
            and text_en
            and samples["audio_language"][idx] == "en"
        )
        mn_to_mn = (
            (not en_to_both and not mn_to_both)
            and text_mn
            and samples["audio_language"][idx] == "mn"
        )

        if en_to_both:
            possible_instructions = [
                inst for inst in instructions if "en" in inst["audio_languages"]
            ]
        elif mn_to_both:
            possible_instructions = [
                inst for inst in instructions if "mn" in inst["audio_languages"]
            ]
        elif en_to_en:
            possible_instructions = [
                inst
                for inst in instructions
                if "en" in inst["audio_languages"]
                and "en" in inst["output_languages"]
                and not inst["all_output_required"]
            ]
        elif mn_to_mn:
            possible_instructions = [
                inst
                for inst in instructions
                if "mn" in inst["audio_languages"]
                and "mn" in inst["output_languages"]
                and not inst["all_output_required"]
            ]
        else:
            possible_instructions = []

        if not possible_instructions:
            print(f"Skipping sample {idx} due to no valid instructions.")
            continue

        instruction_dict = random.choice(possible_instructions)
        instruction = instruction_dict["instruction"]
        output_languages = instruction_dict["output_languages"]

        # Determine the label based on the chosen instruction and output languages
        label = ""
        if (
            "mn" in output_languages
            and "en" in output_languages
            and text_mn is not None
            and text_en is not None
            and instruction_dict["all_output_required"]
        ):
            if output_languages[0] == "mn":
                label = f"### Mongolian\n\n{text_mn}\n\n---\n\n### English\n\n{text_en}"
            else:
                label = f"### English\n\n{text_en}\n\n---\n\n### Mongolian\n\n{text_mn}"
        elif text_mn is not None and text_en is not None:
            if not instruction_dict["all_output_required"]:
                if audio_language in output_languages:
                    label = text_en if audio_language == "en" else text_mn
                elif "mn" in output_languages:
                    label = text_mn
                elif "en" in output_languages:
                    label = text_en
                else:
                    label = text_mn if audio_language == "mn" else text_en
        elif "mn" in output_languages and text_mn is not None:
            label = text_mn
        elif "en" in output_languages and text_en is not None:
            label = text_en
        else:
            print(
                f"Warning: Could not determine label for sample {idx} with instruction {instruction}"
            )
            continue

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": instruction},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples


print("\n---\nFormatting the dataset...\n---\n")

mbspeech = mbspeech.map(format_mix_data, batched=True, batch_size=16, num_proc=8)
ted_en = ted_en.map(format_mix_data, batched=True, batch_size=16, num_proc=8)
ted_mn = ted_mn.map(format_mix_data, batched=True, batch_size=16, num_proc=8)
common_voice = common_voice.map(
    format_mix_data, batched=True, batch_size=16, num_proc=8
)

dataset = DatasetDict()
dataset["train"] = concatenate_datasets(
    [
        mbspeech["train"],
        ted_en["train"],
        ted_mn["train"],
        common_voice["train"],
    ]
).shuffle(seed=RANDOM_SEED)
dataset["test"] = concatenate_datasets(
    [
        mbspeech["test"],
        ted_en["test"],
        ted_mn["test"],
        common_voice["test"],
    ]
).shuffle(seed=RANDOM_SEED)
print(dataset)

example = dataset["train"][144]
print(
    example["audio_language"],
    bool(example["text"]),
    bool(example["text_mn"]),
    bool(example["text_en"]),
)
print(example["messages"][0]["content"][1])
print(example["messages"][1])

example = dataset["test"][452]
print(
    example["audio_language"],
    bool(example["text"]),
    bool(example["text_mn"]),
    bool(example["text_en"]),
)
print(example["messages"][0]["content"][1])
print(example["messages"][1])

print("\n---\nLoading the model...\n---\n")

model = Gemma3nForConditionalGeneration.from_pretrained(
    "google/gemma-3n-E2B-it",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-3n-E2B-it",
    trust_remote_code=True,
)
processor.tokenizer.padding_side = "right"


def collate_fn(examples):
    texts = []
    audios = []

    for example in examples:
        text = processor.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).strip()
        texts.append(text)

        audios.append(example["audio"]["array"])

    batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()

    labels[labels == processor.tokenizer.pad_token_id] = -100
    if hasattr(processor.tokenizer, "image_token_id"):
        labels[labels == processor.tokenizer.image_token_id] = -100
    if hasattr(processor.tokenizer, "audio_token_id"):
        labels[labels == processor.tokenizer.audio_token_id] = -100
    if hasattr(processor.tokenizer, "boi_token_id"):
        labels[labels == processor.tokenizer.boi_token_id] = -100
    if hasattr(processor.tokenizer, "eoi_token_id"):
        labels[labels == processor.tokenizer.eoi_token_id] = -100

    batch["labels"] = labels
    return batch


print("\n---\nTraining...\n---\n")

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,
    # target_modules=["q_proj", "k_proj"],
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.00,
    bias="none",
    use_rslora=False,
    use_dora=False,
    modules_to_save=None,
)

training_args = SFTConfig(
    output_dir="gemma-3n-E2B-it-audio-en-mn",
    eval_strategy="steps",
    eval_steps=300,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-04,
    num_train_epochs=2.0,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    report_to=["tensorboard"],
    run_name="gemma-3n-E2B-it-audio-en-mn",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    max_length=None,
    seed=RANDOM_SEED,
    push_to_hub=True,
)

train_dataset = dataset["train"]
val_dataset = dataset["test"]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset if training_args.eval_strategy != "no" else None,
    processing_class=processor.tokenizer,
    peft_config=peft_config,
)

trainer.train()

trainer.push_to_hub()
