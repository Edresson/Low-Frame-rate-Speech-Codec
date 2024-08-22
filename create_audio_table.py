import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import pandas as pd
import os

import random
from glob import glob
from tqdm import tqdm

import numpy as np
# set seed to ensures reproducibility
def set_seed(random_seed=1234):
    # set deterministic inference
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._set_graph_executor_optimize(False)

set_seed()


def create_html_table(used_samples):
    useful_columns = ["Speaker Name", "Model Name", "generated_wav"]
    df = pd.DataFrame.from_dict(used_samples, orient='columns')
    # Add references as a model
    speaker_references = df["speaker_reference"].unique()

    aux_list = []
    for ref in speaker_references:
        d = {"Speaker Name": "_".join(ref.split("/")[-1].split("_")[:2]), "Model Name": "0 Speaker Reference", "generated_wav": ref}
        aux_list.append(d)

    # drop useless columns
    df.drop(df.columns.difference(useful_columns), 1, inplace=True)

    df = pd.concat([pd.DataFrame.from_dict(aux_list, orient='columns'), df], ignore_index=True)



    # df = df.sort_values('Model Name')
  
    html = df.pivot_table(values=['generated_wav'], index=["Model Name"], columns=['Speaker Name'], aggfunc='sum').to_html()

    # added audio 
    html = html.replace("<td>audios_demo/", '<td><audio controls style="width: 110px;" src="audios_demo/')
    html = html.replace(".wav</td>", '.wav"></audio></td>').replace("Speaker", "Speaker")

    for key in MAP_names:
        model_name = MAP_names[key]
        html = html.replace(model_name, model_name[2:])

    print(html)


EVAL_PATH = "audios_demo/Evaluation/"

samples_files = list(glob(f'{EVAL_PATH}/**/custom_generated_sentences.csv', recursive=True))
samples_files.sort()


MAP_names = {"Speaker Reference": "0 Speaker Reference", 
             "Tortoise": "1 Tortoise",
             "StyleTTS2": "2 StyleTTS 2",
             "Mega-TTS2": "3 Mega-TTS 2",
             "HierSpeech++": "4 HierSpeech++",
             "YourTTS_original": "5 Original YourTTS",
             "YourTTS_libritts_en": "6 YourTTS LibriTTS (Exp 1.)",
             "YourTTS_XTTS": "7 YourTTS XTTS (Exp 2.)",
             "XTTS_v2.0.2":"8 XTTS (Exp 3.)"
}


out_csv = f"samples_demo.csv"




Supported_languages = [
        "en",
        "zh-cn",
        "pt",
        "es",
        "fr",
        "de",
        "it",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "hu",
        "ko",
        "ja"
]

lang_map = {

        "en": "English",
        "pt": "Portuguese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
        "nl": "Dutch",
        "cs": "Czech",
        "ar": "Arabic",
        "zh-cn": "Chinese",
        "hu": "Hungarian",
        "ko": "Korean",
        "ja": "Japanese"
}

all_used_samples = []
for lang in Supported_languages:
    selected_audio_paths = {}
    selected_text = {}
    used_samples = []

    for SAMPLES_CSV in samples_files:
        if "FT/" in SAMPLES_CSV:
            continue
        df = pd.read_csv(SAMPLES_CSV)

        df = df.loc[df['language'].isin([lang])]
        if df.empty:
            continue
        # group by speaker_reference
        df_speakers = df.groupby('speaker_reference')
        for name, df_speaker in df_speakers:
            # if sample was already selected for this speaker find from anothers DF
            if selected_audio_paths and name in selected_audio_paths.keys():
                selected_item = df_speaker.loc[df_speaker['text'].isin(selected_text[name])].iloc[0]
                selected_item["Speaker Name"] = "_".join(name.split("/")[-1].split("_")[:2])
                selected_item["Model Name"] = SAMPLES_CSV.split("/Evaluation/")[-1].split("/")[0]

                for key in MAP_names:
                    if key in selected_item["Model Name"]:
                        selected_item["Model Name"] = MAP_names[key]
                        break

                selected_item["generated_wav"] = os.path.join(EVAL_PATH.replace("Evaluation/", ""), selected_item["generated_wav"])
                selected_item["speaker_reference"] = os.path.join(EVAL_PATH.replace("Evaluation/", ""), selected_item["speaker_reference"])
                selected_audio_paths[name].append(selected_item["generated_wav"])
                # selected_text[name].append(selected_item["text"])
                used_samples.append(selected_item)
            else:
                # random select a sentence
                selected_item = df_speaker.sample(n=1).iloc[0]
                selected_item["Speaker Name"] = "_".join(name.split("/")[-1].split("_")[:2])
                selected_item["Model Name"] = SAMPLES_CSV.split("audios_demo/Evaluation/")[-1].split("/")[0]
                selected_item["generated_wav"] = os.path.join(EVAL_PATH.replace("Evaluation/", ""), selected_item["generated_wav"])
                selected_item["speaker_reference"] = os.path.join(EVAL_PATH.replace("Evaluation/", ""), selected_item["speaker_reference"])
                
                for key in MAP_names:
                    if key in selected_item["Model Name"]:
                        selected_item["Model Name"] = MAP_names[key]
                        break

                # print(selected_item)
                selected_audio_paths[name] = [selected_item["generated_wav"]]
                selected_text[name] = [selected_item["text"]]
                used_samples.append(selected_item)
    all_used_samples.extend(used_samples)
    language = lang_map[lang]
    print(f"\n\n <p><b>{language} samples</b></p>\n")
    create_html_table(used_samples)


# print(selected_audio_paths)



df_out = pd.DataFrame.from_dict(all_used_samples, orient='columns')
df_out.to_csv(out_csv, index=False, sep=',', encoding='utf-8')
print("CSV saved at:", out_csv)

# delete all unused audio samples
used_files = df_out["generated_wav"].tolist()
# print(used_files)
audio_files = list(glob(f'{EVAL_PATH}/**/*.wav', recursive=True))
# print(audio_files[:10])

for file in audio_files:
    if file not in used_files:
        # print(file)
        if os.path.isfile(file):
            os.remove(file)
# df_out["generated_wav"]