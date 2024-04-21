import os
import sys
import shutil
import zipfile
import urllib.request
from urllib.parse import urlparse, unquote
from argparse import Namespace
from cog import BasePredictor, Input, Path as CogPath

sys.path.insert(0, os.path.abspath("src"))
import main as m

def download_online_model(url, dir_name):
    print(f"[~] Downloading voice model with name {dir_name}...")
    zip_name = os.path.basename(urlparse(url).path)
    extraction_folder = os.path.join(m.rvc_models_dir, dir_name)
    if os.path.exists(extraction_folder):
        print(f"Voice model directory {dir_name} already exists! Skipping download.")
        return

    urllib.request.urlretrieve(url, zip_name)
    print("[~] Extracting zip...")
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(extraction_folder)
    print(f"[+] {dir_name} Model successfully downloaded!")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """ Setup or load the model into memory """
        pass

    def predict(self, song_input: CogPath = Input(description="Upload your audio file here."),
                rvc_model: str = Input(description="RVC model for a specific voice.", default="Squidward",
                                       choices=["Squidward", "MrKrabs", "Plankton", "Drake", "Vader", "Trump", "Biden",
                                                "Obama", "Guitar", "Voilin", "CUSTOM"]),
                custom_rvc_model_download_url: str = Input(description="URL to download a custom RVC model.",
                                                           default=None),
                pitch_change: str = Input(description="Adjust pitch of AI vocals.", default="no-change",
                                          choices=["no-change", "male-to-female", "female-to-male"]),
                index_rate: float = Input(description="Control how much of the AI's accent to leave in the vocals.",
                                          default=0.5),
                filter_radius: int = Input(description="Apply median filtering to the harvested pitch results.",
                                           default=3),
                rms_mix_rate: float = Input(description="Control how much to use the original vocal's loudness.",
                                            default=0.25),
                pitch_detection_algorithm: str = Input(description="Select pitch detection algorithm.",
                                                       default="rmvpe",
                                                       choices=["rmvpe", "mangio-crepe"]),
                crepe_hop_length: int = Input(description="Controls how often it checks for pitch changes in milliseconds.",
                                              default=128),
                protect: float = Input(description="Control protection of voiceless consonants and breath sounds.",
                                       default=0.33),
                main_vocals_volume_change: float = Input(description="Volume change for AI main vocals.",
                                                         default=0),
                instrumental_volume_change: float = Input(description="Volume change for the background music.",
                                                          default=0),
                pitch_change_all: float = Input(description="Change pitch/key of all audio components in semitones.",
                                                default=0),
                output_format: str = Input(description="Output format of audio file.", default="mp3",
                                           choices=["mp3", "wav"])):
        if custom_rvc_model_download_url:
            dir_name = unquote(custom_rvc_model_download_url.split("/")[-1]).split('.')[0]
            download_online_model(custom_rvc_model_download_url, dir_name)
            rvc_model = dir_name

        pitch_value = {"no-change": 0, "male-to-female": 1, "female-to-male": -1}[pitch_change]
        args = Namespace(song_input=str(song_input), rvc_dirname=rvc_model, pitch_change=pitch_value,
                         index_rate=index_rate, filter_radius=filter_radius, rms_mix_rate=rms_mix_rate,
                         pitch_detection_algo=pitch_detection_algorithm, crepe_hop_length=crepe_hop_length,
                         protect=protect, main_vol=main_vocals_volume_change, inst_vol=instrumental_volume_change,
                         pitch_change_all=pitch_change_all, output_format=output_format, keep_files=False)

        output_path = m.song_cover_pipeline(args)
        print(f'[+] Cover generated at {output_path}')
        return CogPath(output_path)
