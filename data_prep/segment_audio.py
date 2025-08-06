import argparse
import os
import sys
from pathlib import Path

import srt
from pydub import AudioSegment


def parse_srt_file(srt_path):
    """Parse SRT file using the srt library and return list of subtitle entries."""
    subtitles = []

    with open(srt_path, "r", encoding="utf-8") as f:
        subtitle_generator = srt.parse(f.read())

        for subtitle in subtitle_generator:
            start_ms = int(subtitle.start.total_seconds() * 1000)
            end_ms = int(subtitle.end.total_seconds() * 1000)

            subtitles.append(
                {
                    "number": subtitle.index,
                    "start": start_ms,
                    "end": end_ms,
                    "text": subtitle.content,
                }
            )

    return subtitles


def merge_subtitles_to_segments(subtitles_en, subtitles_mn, max_duration_ms):
    """Merge multiple subtitle items to maximize duration under the limit."""
    segments = []
    i = 0

    while i < len(subtitles_en):
        # Start a new segment
        segment_start = subtitles_en[i]["start"]
        segment_end = subtitles_en[i]["end"]
        en_texts = [subtitles_en[i]["text"]]
        mn_texts = [subtitles_mn[i]["text"]]
        subtitle_indices = [i]

        # Try to merge subsequent subtitles
        j = i + 1
        while j < len(subtitles_en):
            next_end = subtitles_en[j]["end"]
            potential_duration = next_end - segment_start

            # Check if adding this subtitle would exceed max duration
            if potential_duration <= max_duration_ms:
                segment_end = next_end
                en_texts.append(subtitles_en[j]["text"])
                mn_texts.append(subtitles_mn[j]["text"])
                subtitle_indices.append(j)
                j += 1
            else:
                break

        # Create segment
        segments.append(
            {
                "start": segment_start,
                "end": segment_end,
                "en_text": " ".join(en_texts).replace("\n", " ").strip(),
                "mn_text": " ".join(mn_texts).replace("\n", " ").strip(),
                "subtitle_indices": subtitle_indices,
            }
        )

        i = j

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Split audio files using SRT files with duration limits"
    )
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("srt_en", help="Path to English SRT file")
    parser.add_argument("srt_mn", help="Path to Mongolian SRT file")
    parser.add_argument("output_dir", help="Output directory for split files")
    parser.add_argument(
        "--max-duration",
        type=int,
        default=30,
        help="Maximum duration for each split in seconds (default: 30)",
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    if not os.path.exists(args.srt_en):
        print(f"Error: English SRT file not found: {args.srt_en}")
        sys.exit(1)

    if not os.path.exists(args.srt_mn):
        print(f"Error: Mongolian SRT file not found: {args.srt_mn}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio file: {args.audio_file}")

    # Load audio (assuming it's already in WAV format)
    try:
        audio = AudioSegment.from_wav(args.audio_file)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)

    print("Parsing SRT files...")

    # Parse SRT files
    try:
        subtitles_en = parse_srt_file(args.srt_en)
        subtitles_mn = parse_srt_file(args.srt_mn)
    except Exception as e:
        print(f"Error parsing SRT files: {e}")
        sys.exit(1)

    # Verify both SRT files have same number of subtitles
    if len(subtitles_en) != len(subtitles_mn):
        print(
            f"Warning: SRT files have different number of subtitles ({len(subtitles_en)} vs {len(subtitles_mn)})"
        )

    max_duration_ms = args.max_duration * 1000

    # Get the base name of the audio file (without extension)
    audio_basename = Path(args.audio_file).stem

    print(f"Merging subtitles with max duration: {args.max_duration} seconds")

    # Merge subtitles into optimal segments
    segments = merge_subtitles_to_segments(subtitles_en, subtitles_mn, max_duration_ms)

    print(f"Created {len(segments)} segments from {len(subtitles_en)} subtitles")

    # Process each merged segment
    for segment_idx, segment in enumerate(segments):
        segment_count = segment_idx + 1

        # Extract audio segment
        audio_segment = audio[segment["start"] : segment["end"]]
        print(len(audio), segment["start"], segment["end"])

        # Generate output filenames
        base_name = f"{audio_basename}_{segment_count:04d}"

        wav_file = output_dir / f"{base_name}.wav"
        en_text_file = output_dir / f"{base_name}.en.txt"
        mn_text_file = output_dir / f"{base_name}.mn.txt"

        # Export audio segment
        audio_segment.export(str(wav_file), format="wav")

        # Write text files
        with open(en_text_file, "w", encoding="utf-8") as f:
            f.write(segment["en_text"])

        with open(mn_text_file, "w", encoding="utf-8") as f:
            f.write(segment["mn_text"])

        duration_sec = (segment["end"] - segment["start"]) / 1000
        subtitle_range = f"{min(segment['subtitle_indices']) + 1}-{max(segment['subtitle_indices']) + 1}"
        print(
            f"Created: {base_name}.wav ({duration_sec:.2f}s) from subtitles {subtitle_range}"
        )

    print(f"\nCompleted! Created {len(segments)} audio segments in {output_dir}")


if __name__ == "__main__":
    main()
