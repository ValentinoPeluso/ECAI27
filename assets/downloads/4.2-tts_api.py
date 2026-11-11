import base64
import string

import litserve as ls
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class TTSLitAPI(ls.LitAPI):
    def setup(self, device):
        model_name = 'openai/whisper-tiny.en'
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name
        )

    def decode_request(self, request):
        raw = base64.b64decode(request['input'])
        x = torch.frombuffer(raw, dtype=torch.float32)

        return {'x': x, 'sampling_rate': request['sampling_rate']}

    def predict(self, inputs):
        input_features = self.processor(
            inputs['x'],
            sampling_rate=inputs['sampling_rate'],
            return_tensors='pt',
        ).input_features
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False
        )
        transcription = (
            transcription[0]
            .strip()
            .translate(str.maketrans('', '', string.punctuation))
            .lower()
        )

        return transcription

    def encode_response(self, output):
        return {'output': output}


if __name__ == '__main__':
    api = TTSLitAPI()
    server = ls.LitServer(
        api, accelerator='cpu', devices=1, workers_per_device=1
    )
    server.run(port=8000)
