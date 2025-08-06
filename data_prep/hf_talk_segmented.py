import os

from datasets import Audio, Dataset, DatasetDict


def create_audio_dataset(audio_dir, en_dir, mn_dir, lang):
    filenames = []
    groups = []
    audio_files = []
    english_texts = []
    mongolian_texts = []
    audio_languages = []

    for filename in sorted(os.listdir(audio_dir)):
        if filename.endswith((".wav", ".mp3", ".flac", ".m4a")):
            audio_path = os.path.join(audio_dir, filename)
            base_name = os.path.splitext(filename)[0]
            en_sub_path = os.path.join(en_dir, f"{base_name}.en.txt")
            mn_sub_path = os.path.join(mn_dir, f"{base_name}.mn.txt")

            if os.path.exists(en_sub_path):
                audio_files.append(audio_path)
                en_sub_text = ""
                mn_sub_text = ""
                with open(en_sub_path, "r", encoding="utf-8") as f:
                    en_sub_text = f.read().strip()
                with open(mn_sub_path, "r", encoding="utf-8") as f:
                    mn_sub_text = f.read().strip()

                english_texts.append(en_sub_text)
                mongolian_texts.append(mn_sub_text)
                filenames.append(filename)
                groups.append(base_name[:-5])
                audio_languages.append(lang)
            else:
                print(f"Warning: English subtitle file not found for {filename}")

    data = {
        "audio": audio_files,
        "text_en": english_texts,
        "text_mn": mongolian_texts,
        "filename": filenames,
        "group": groups,
        "audio_language": audio_languages,
    }

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio())

    return dataset


hf_token = os.getenv("HF_TOKEN", "")
dataset_repo_name = "bilguun/ted_talks_en_mn_segmented"

audio_directory = "talk/segmented"
en_directory = "talk/segmented"
mn_directory = "talk/segmented"

dataset_en = create_audio_dataset(
    audio_directory, en_directory, mn_directory, lang="en"
)

print("English dataset created with", len(dataset_en), "entries.")
print("Sample entry:", dataset_en[0])

audio_directory = "talks_mn/segmented"
en_directory = "talks_mn/segmented"
mn_directory = "talks_mn/segmented"

dataset_mn = create_audio_dataset(
    audio_directory, en_directory, mn_directory, lang="mn"
)

print("Mongolian dataset created with", len(dataset_mn), "entries.")
print("Sample entry:", dataset_mn[0])

dataset = DatasetDict({"en": dataset_en, "mn": dataset_mn})

dataset.push_to_hub(dataset_repo_name, private=True, token=hf_token)
