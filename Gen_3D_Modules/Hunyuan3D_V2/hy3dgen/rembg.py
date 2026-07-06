# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image
from rembg import remove, new_session
import numpy as np
import os


class BackgroundRemover():
    def __init__(self, model_name='u2net', model_path=None):
        if model_path:
            model_path = os.path.abspath(os.path.expanduser(model_path))
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"rembg model file not found: {model_path}")
            cache_dir = os.path.expanduser("~/.u2net")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{model_name}.onnx")
            if os.path.exists(cache_path) or os.path.islink(cache_path):
                if os.path.realpath(cache_path) != os.path.realpath(model_path):
                    os.unlink(cache_path)
            if not os.path.exists(cache_path):
                os.symlink(model_path, cache_path)
        self.session = new_session(model_name)

    def __call__(self, image: Image.Image):
        if image.mode in ('RGBA', 'LA') or image.mode != 'RGB':
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background
        
        output = remove(image, session=self.session)
        
        if output.mode != 'RGBA':
            output = output.convert('RGBA')
            
        return output
