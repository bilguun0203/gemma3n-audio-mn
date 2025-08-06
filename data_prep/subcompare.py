import os
from datetime import timedelta

import srt

# sub offset 3:550
# en_sub_path = "talk/englishsub/ How to change your behavior for the better.en.srt"
# mn_sub_path = "talk/mongoliasub/ How to change your behavior for the better.mn.srt"

sub_file_paths = sorted(os.listdir("talk/englishsub"))

for sub_path in sub_file_paths:
    en_sub_path = os.path.join("talk/englishsub", sub_path)
    if not en_sub_path.endswith(".en.srt"):
        continue
    mn_sub_path = os.path.join(
        "talk/mongoliasub", sub_path.replace(".en.srt", ".mn.srt")
    )

    if not os.path.exists(mn_sub_path):
        print(f"Missing Mongolian subtitle for {en_sub_path}")
        continue

    with open(en_sub_path, "r") as f:
        en_sub = srt.parse(f.read())
    with open(mn_sub_path, "r") as f:
        mn_sub = srt.parse(f.read())

    en_sub = [sub for sub in en_sub]
    mn_sub = [sub for sub in mn_sub]

    if len(en_sub) != len(mn_sub):
        print("diff")
    print(len(en_sub), len(mn_sub))

    # for en, mn in zip(en_sub, mn_sub):
    #     print(f"{en.index}\t{en.start}->{en.end}\t{mn.start}->{mn.end}")

    mn_sub_index = 0
    ACCEPTABLE_DIFF = timedelta(milliseconds=50)
    fixed_mn_sub = []

    for en in en_sub:
        best_match_index = None
        best_time_diff = None

        for i, mn in enumerate(mn_sub):
            time_diff = abs(en.start - mn.start)

            if best_time_diff is None or time_diff < best_time_diff:
                best_time_diff = time_diff
                best_match_index = i

        if best_match_index is not None:
            mn_match = mn_sub[best_match_index]

            if best_time_diff <= ACCEPTABLE_DIFF:
                fixed_subtitle = srt.Subtitle(
                    index=en.index, start=en.start, end=en.end, content=mn_match.content
                )
                fixed_mn_sub.append(fixed_subtitle)
                print(
                    f"Aligned #{en.index}: EN {en.start}-{en.end} with MN {mn_match.start}-{mn_match.end} (diff: {best_time_diff})"
                )
            else:
                fixed_subtitle = srt.Subtitle(
                    index=en.index,
                    start=mn_match.start,
                    end=mn_match.end,
                    content=mn_match.content,
                )
                fixed_mn_sub.append(fixed_subtitle)
                print(f"Kept original #{en.index}: Large time diff {best_time_diff}")

    output_path = f"talk/mnsub/{sub_path.replace('.en.srt', '.mn.srt')}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(fixed_mn_sub))

    print(f"\nFixed subtitles saved to: {output_path}")
    print(f"Processed {len(fixed_mn_sub)} subtitles")

