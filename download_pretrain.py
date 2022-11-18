import os
import time
import gdown

zip_path = "tts_en_talknet_1.0.0rc1.zip"
for i in range(10):
    if not os.path.exists(zip_path) or os.stat(zip_path).st_size < 100:
        gdown.download(
                "https://drive.google.com/uc?id=19wSym9mNEnmzLS9XdPlfNAW9_u-mP1hR",
                zip_path,
                quiet=False,
        )

