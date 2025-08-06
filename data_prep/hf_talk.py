import os

import srt
from datasets import Audio, Dataset, DatasetDict


def create_audio_dataset(audio_dir, en_dir, mn_dir, lang):
    filenames = []
    audio_files = []
    english_subs = []
    english_texts = []
    mongolian_subs = []
    mongolian_texts = []
    audio_languages = []

    for filename in sorted(os.listdir(audio_dir)):
        if filename.endswith((".wav", ".mp3", ".flac", ".m4a")):
            audio_path = os.path.join(audio_dir, filename)
            base_name = os.path.splitext(filename)[0]
            en_sub_path = os.path.join(en_dir, f"{base_name}.en.srt")
            mn_sub_path = os.path.join(mn_dir, f"{base_name}.mn.srt")

            if os.path.exists(en_sub_path):
                audio_files.append(audio_path)
                en_sub_text = ""
                mn_sub_text = ""
                with open(en_sub_path, "r", encoding="utf-8") as f:
                    en_sub_content = f.read().strip()
                with open(mn_sub_path, "r", encoding="utf-8") as f:
                    mn_sub_content = f.read().strip()

                if en_sub_content:
                    en_sub_text = "\n".join(
                        [s.content for s in srt.parse(en_sub_content)]
                    )
                if mn_sub_content:
                    mn_sub_text = "\n".join(
                        [s.content for s in srt.parse(mn_sub_content)]
                    )

                english_subs.append(en_sub_content)
                mongolian_subs.append(mn_sub_content)
                english_texts.append(en_sub_text)
                mongolian_texts.append(mn_sub_text)
                filenames.append(filename)
                audio_languages.append(lang)

    data = {
        "audio": audio_files,
        "subtitle_en": english_subs,
        "subtitle_mn": mongolian_subs,
        "text_en": english_texts,
        "text_mn": mongolian_texts,
        "filename": filenames,
        "audio_language": audio_languages,
    }

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio())

    return dataset


hf_token = os.getenv("HF_TOKEN", "")
dataset_repo_name = "bilguun/ted_talks_en_mn"

audio_directory = "talk/wavs"
en_directory = "talk/ensub_fixed"
mn_directory = "talk/mnsub_fixed"

dataset_en = create_audio_dataset(
    audio_directory, en_directory, mn_directory, lang="en"
)

print("English dataset created with", len(dataset_en), "entries.")
print("Sample entry:", dataset_en[0])

audio_directory = "talks_mn/wavs"
en_directory = "talks_mn/ensub"
mn_directory = "talks_mn/mnsub"

dataset_mn = create_audio_dataset(
    audio_directory, en_directory, mn_directory, lang="mn"
)

print("Mongolian dataset created with", len(dataset_mn), "entries.")
print("Sample entry:", dataset_mn[0])

dataset = DatasetDict({"en": dataset_en, "mn": dataset_mn})

dataset.push_to_hub(dataset_repo_name, private=True, token=hf_token)
