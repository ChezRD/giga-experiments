"""Utils for loading datasets from hub."""

import enum
from typing import Optional, Union, List, Tuple

import pandas as pd
from datasets import load_dataset


class DatasetsAvailable(enum.Enum):
    """Datasets available"""

    MultidomainGold = "Multidomain gold dataset. For more see `ai-forever/spellcheck_punctuation_benchmark`."
    RUSpellRU = "Social media texts and blogs. For more see `ai-forever/spellcheck_punctuation_benchmark`."
    MedSpellchecker = "Medical anamnesis. For more see `ai-forever/spellcheck_punctuation_benchmark`."
    GitHubTypoCorpusRu = "Github commits. For more see `ai-forever/spellcheck_punctuation_benchmark`."

    MultidomainGold_orth = "Multidomain gold dataset orthography only. For more see `ai-forever/spellcheck_benchmark`."
    RUSpellRU_orth = "Social media texts and blogs orthography only. For more see `ai-forever/spellcheck_benchmark`."
    MedSpellchecker_orth = "Medical anamnesis orthography only. For more see `ai-forever/spellcheck_benchmark`."
    GitHubTypoCorpusRu_orth = "Github commits orthography only. For more see `ai-forever/spellcheck_benchmark`."


datasets_available = [dataset.name for dataset in DatasetsAvailable]


def load_available_dataset_from_hf(
        dataset_name: str, for_labeler: bool, split: Optional[str] = None
) -> Union[Tuple[List[str], List[str]], pd.DataFrame]:
    if dataset_name not in datasets_available:
        raise ValueError("You provided wrong dataset name: {}\nAvailable datasets are: {}".format(
            dataset_name, *datasets_available))
    source_collection = "spellcheck_punctuation_benchmark"
    if dataset_name[-4:] == "orth":
        source_collection = "spellcheck_benchmark"
        dataset_name = dataset_name[:-5]
    dataset = load_dataset("ai-forever/{}".format(source_collection), dataset_name, split=split)
    if split is None:
        dataset = pd.concat([dataset[split].to_pandas() for split in dataset.keys()]).reset_index(drop=True)
    else:
        dataset = dataset.to_pandas()
    if for_labeler:
        sources = dataset.source.values.tolist()
        corrections = dataset.correction.values.tolist()
        return sources, corrections
    return dataset
