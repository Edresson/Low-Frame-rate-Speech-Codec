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


def create_html_table(dic):


    
    df = pd.DataFrame.from_dict(dic, orient='columns')
    # print(df)
    # df = pd.concat([pd.DataFrame.from_dict(aux_list, orient='columns'), df], ignore_index=True)
    # df = df.sort_values('Model Name')
    
    # html = df.pivot_table(values=['generated_wav'], index=["Model Name"], columns=['Speaker Name'], aggfunc='sum').to_html()

    html = df.pivot_table(values=['Samples'], index=["Codec"], columns=['Speaker Name'], aggfunc='sum').to_html()

    # added audio 
    html = html.replace("<td>audios_demo/", '<td><audio controls style="width: 110px;" src="audios_demo/')
    html = html.replace(".wav</td>", '.wav"></audio></td>')
    for key in model_map:
        html = html.replace(model_map[key], key)

    html  = html.replace("@", "")

    print(html)









lang_map = {

        "en": "English",
        "pt": "Portuguese",
        "sp": "Spanish",
        "fr": "French",
        "ge": "German",
        "it": "Italian",
        "pl": "Polish",
        "du": "Dutch",
}

samples_path = "/home/ecasanova/Projects/Papers/ICASSP-2025-21Hz-codec/NeMo-Speech-Codec/audios_demo/codecs_reconstruction_22kHz/"

from glob import glob


all_samples = glob(samples_path + '**/*.wav', recursive=True)
all_samples.sort()
model_map = {
    "GT": "0 Ground truth",
    "Encodec 6kbps": "1 Encodec 6kbps",
    "DAC 7.75kbps": "2 DAC 7.75kbps",
    "Spectral Codec": "3 Spectral Codec",
    "Ours 2k codes": "4 Ours 2k codes",
    "Ours 4k codes": "5 Ours 4k codes",
}

language_samples = {}
count_daps = set()
for sample in all_samples:
    if "/daps/" in sample:
        lang = "DAPS"
    else:
        lang = os.path.dirname(sample).split("/")[-1]
        lang = lang_map[lang]
    
    audio_path = sample.split("NeMo-Speech-Codec/")[-1]

    model_name = os.path.basename(audio_path).split("_")[-1].replace(".wav", "").replace("-", " ").replace("SpectralCodec", "Spectral Codec")
    speaker_name = os.path.basename(audio_path).split("_")[0]

    if "/daps/" in sample:
        sample_base_name = "_".join(os.path.basename(audio_path).split("_")[:-1])
        count_daps.add(sample_base_name)

        speaker_name = speaker_name+" "+ ("@" * len(count_daps))
        # print(speaker_name)
    dic = {"Codec": model_map[model_name], "Samples": audio_path, "Speaker Name": speaker_name}

    
    if lang not in language_samples:
        language_samples[lang] = [dic]
    else:
        language_samples[lang].append(dic)


for language in language_samples:
    print(f"\n\n <p><b>{language} samples</b></p>\n")
    create_html_table(language_samples[language])
