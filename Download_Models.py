# Download the Llama-3.1-8B-Omni model from Huggingface.
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ICTNLP/Llama-3.1-8B-Omni", local_dir="models/Llama-3.1-8B-Omni/", revision="main")

# Download the Whisper-large-v3 model.
import whisper
model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
