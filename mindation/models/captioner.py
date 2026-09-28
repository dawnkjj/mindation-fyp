#image-to-text model wrapper for lecture screenshots and whiteboards
from __future__ import annotations

# library imports
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from mindation.config import ModelSettings

logger = logging.getLogger(__name__)

# load blip model so can run without heavy ai
@lru_cache(maxsize=1)
def _load_blip(model_name: str) -> tuple[Any, Any]:

    from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model

# upload slide whiteboard images using blip
class ImageCaptioner:

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    #generate caption for an image
    def caption(self, image_path: Path) -> tuple[str, str | None]:

        if not image_path.exists():
            return "", f"Image file not found: {image_path}"
        if not self.settings.use_blip_captioning:
            return "", "Image captioning is disabled in the sidebar settings."
        try:
            from PIL import Image
            import torch

            # load the BLIP model and processor, then generate a caption for the image
            processor, model = _load_blip(self.settings.image_caption_model)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=45)
            caption = processor.decode(output[0], skip_special_tokens=True).strip()
            return caption, None if caption else "BLIP returned an empty image caption."

        # handle missing dependencies or other errors
        except ImportError:
            return "", "BLIP captioning dependencies are missing. Run: python -m pip install -r requirements-ai.txt"
        except ModuleNotFoundError:
            return "", "BLIP captioning dependencies are missing. Run: python -m pip install -r requirements-ai.txt"
        except Exception as exc:
            logger.exception("BLIP captioning failed")
            return "", f"Image captioning failed for {image_path.name}: {exc}"


