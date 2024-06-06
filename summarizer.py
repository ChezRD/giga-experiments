# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# MODEL_REPO = "RussianNLP/FRED-T5-Summarizer"
MODEL_REPO = "cointegrated/rut5-base-absum"

class Summarizer():
    def __init__(self, device):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, legacy=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
        self.model = self.model.to(device)
        self.model.eval()
        
    def summarize(self, text, n_words=None, compression=None, max_length=6000, num_beams=3, do_sample=False, repetition_penalty=10.0, **kwargs):
        """
        Summarize the text
        The following parameters are mutually exclusive:
        - n_words (int) is an approximate number of words to generate.
        - compression (float) is an approximate length ratio of summary and original text.
        """
        if n_words:
            text = '[{}] '.format(n_words) + text
        elif compression:
            text = '[{0:.1g}] '.format(compression) + text
        x = self.tokenizer(text, return_tensors='pt', padding=True).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **x, 
                max_length=max_length, num_beams=num_beams, 
                do_sample=do_sample, repetition_penalty=repetition_penalty, 
                **kwargs
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)