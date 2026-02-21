from __future__ import annotations

import typing as t
import time
import datetime
from transformers import StoppingCriteria, StoppingCriteriaList
import bentoml
from PIL.Image import Image

MODEL_ID = "Salesforce/blip-image-captioning-base"

runtime_image = bentoml.images.PythonImage(python_version="3.12")

@bentoml.service(
    image=runtime_image,
    resources={"gpu": 1}
)
class BlipImageCaptioning:
    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID)

    def __init__(self) -> None:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.bfloat16

        self.model = BlipForConditionalGeneration.from_pretrained(self.hf_model).to(self.device)
        self.processor = BlipProcessor.from_pretrained(self.hf_model)
        print("Model blip loaded on", self.device)

    class FirstTokenTimingCallback(StoppingCriteria):
        def __init__(self):
            self.first_token_time: float | None = None
            self.first_token_datetime: datetime.datetime | None = None
            self.start_time = time.time()
            self.start_datetime = datetime.datetime.now()

        def __call__(self, input_ids, scores, **kwargs):
            if self.first_token_time is None:
                # record both elapsed seconds and wall-clock time
                self.first_token_time = time.time() - self.start_time
                self.first_token_datetime = datetime.datetime.now()
            return False

    @bentoml.api
    async def generate(self, img: Image, txt: t.Optional[str] = None) -> t.Dict[str, t.Any]:
        # preprocessing (same as before)…
        img = img.convert("RGB")
        preprocess_start = time.time()
        if txt:
            inputs = self.processor(img, txt, return_tensors="pt").to(self.device, self.dtype)
        else:
            inputs = self.processor(img, return_tensors="pt").to(self.device, self.dtype)
        preprocess_time = time.time() - preprocess_start

        # --- generation timing & datetimes ---
        # 1) mark wall-clock and perf start
        callback = self.FirstTokenTimingCallback()
        generation_start = time.time()
        generation_start_datetime = callback.start_datetime

        # 2) run generate with callback
        out = self.model.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=20,
            stopping_criteria=StoppingCriteriaList([callback])
        )
        total_generation_time = time.time() - generation_start

        # 3) mark end time
        end_datetime = datetime.datetime.now()

        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return {
            "caption": caption,
            "preprocessing_time": preprocess_time,
            "generation_start_datetime": generation_start_datetime.isoformat(),
            "time_to_first_token": callback.first_token_time,
            "first_token_datetime": callback.first_token_datetime.isoformat()
                                     if callback.first_token_datetime else None,
            "total_generation_time": total_generation_time,
            "generation_end_datetime": end_datetime.isoformat(),
        }
