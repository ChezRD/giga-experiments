# -*- coding: utf-8 -*-

from transformers import pipeline
from transformers import AutoTokenizer, BertForTokenClassification

def process_token(token, label):
    if label == "LOWER_O":
        return token
    if label == "LOWER_PERIOD":
        return token + "."
    if label == "LOWER_COMMA":
        return token + ","
    if label == "LOWER_QUESTION":
        return token + "?"
    if label == "LOWER_TIRE":
        return token + "—"
    if label == "LOWER_DVOETOCHIE":
        return token + ":"
    if label == "LOWER_VOSKL":
        return token + "!"
    if label == "LOWER_PERIODCOMMA":
        return token + ";"
    if label == "LOWER_DEFIS":
        return token + "-"
    if label == "LOWER_MNOGOTOCHIE":
        return token + "..."
    if label == "LOWER_QUESTIONVOSKL":
        return token + "?!"
    if label == "UPPER_O":
        return token.capitalize()
    if label == "UPPER_PERIOD":
        return token.capitalize() + "."
    if label == "UPPER_COMMA":
        return token.capitalize() + ","
    if label == "UPPER_QUESTION":
        return token.capitalize() + "?"
    if label == "UPPER_TIRE":
        return token.capitalize() + " —"
    if label == "UPPER_DVOETOCHIE":
        return token.capitalize() + ":"
    if label == "UPPER_VOSKL":
        return token.capitalize() + "!"
    if label == "UPPER_PERIODCOMMA":
        return token.capitalize() + ";"
    if label == "UPPER_DEFIS":
        return token.capitalize() + "-"
    if label == "UPPER_MNOGOTOCHIE":
        return token.capitalize() + "..."
    if label == "UPPER_QUESTIONVOSKL":
        return token.capitalize() + "?!"
    if label == "UPPER_TOTAL_O":
        return token.upper()
    if label == "UPPER_TOTAL_PERIOD":
        return token.upper() + "."
    if label == "UPPER_TOTAL_COMMA":
        return token.upper() + ","
    if label == "UPPER_TOTAL_QUESTION":
        return token.upper() + "?"
    if label == "UPPER_TOTAL_TIRE":
        return token.upper() + " —"
    if label == "UPPER_TOTAL_DVOETOCHIE":
        return token.upper() + ":"
    if label == "UPPER_TOTAL_VOSKL":
        return token.upper() + "!"
    if label == "UPPER_TOTAL_PERIODCOMMA":
        return token.upper() + ";"
    if label == "UPPER_TOTAL_DEFIS":
        return token.upper() + "-"
    if label == "UPPER_TOTAL_MNOGOTOCHIE":
        return token.upper() + "..."
    if label == "UPPER_TOTAL_QUESTIONVOSKL":
        return token.upper() + "?!"

MODEL_REPO = "RUPunct/RUPunct_big"
MAX_LENGTH = 512

class RUPunkt():
    def __init__(self, device):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, strip_accents=False, add_prefix_space=True)
        self.classifier = pipeline("ner", device=device, model=MODEL_REPO, tokenizer=self.tokenizer, aggregation_strategy="first")


    def punctuate(self, input_text):
        preds = self.classifier(input_text)
        output = ""
        for item in preds:
            output += " " + process_token(item['word'].strip(), item['entity_group'])

        return output.strip()


        # # Split the input text into chunks
        # tokens = self.tokenizer(input_text, return_tensors="pt", truncation=False, padding=False)["input_ids"][0]
        # chunks = [tokens[i:i + self.max_length] for i in range(0, len(tokens), self.max_length)]

        # output = ""
        # for chunk in chunks:
        #     chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
        #     preds = self.classifier(chunk_text)

        #     for item in preds:
        #         output += " " + process_token(item['word'].strip(), item['entity_group'])

        # return output.strip()