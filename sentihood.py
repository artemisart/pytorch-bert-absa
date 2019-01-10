import json
from collections import Counter
from pathlib import Path

import attr
import torch

from pytorch_pretrained_bert import BertModel
from tokenization import BertTokenizer

MODEL = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(
    MODEL,
    do_lower_case='uncased' in MODEL,
    never_split='[UNK] [SEP] [PAD] [CLS] [MASK] LOCATION1 LOCATION2'.split(),
)

LOCATIONS = ['LOCATION1', 'LOCATION2']
for loc in LOCATIONS:
    tokenizer.vocab[loc] = tokenizer.vocab['[MASK]']
SEP = tokenizer.vocab['[SEP]']

@attr.s(auto_attribs=True, slots=True)
class Example:
    """Represent a training or test example"""

    id: int
    text: str
    segment_ids: [int]
    target: str
    target_idx: int
    aspect: str
    sentiment: str


@attr.s(auto_attribs=True, slots=True)
class Dataset:
    """Represent a dataset (train/text/val) of examples which can be of any type"""

    train: list
    dev: list
    test: list

    def get_all(self):
        return [('train', self.train), ('dev', self.dev), ('test', self.test)]

    def apply(self, func):
        """Apply some function on all the examples"""
        for _, data in self.get_all():
            for example in data:
                func(example)

    def map(self, func):
        return self.__class__(
            train=[func(ex) for ex in self.train],
            dev=[func(ex) for ex in self.dev],
            test=[func(ex) for ex in self.test],
        )

    def map_many(self, func):
        return self.__class__(
            train=[x for ex in self.train for x in func(ex)],
            dev=[x for ex in self.dev for x in func(ex)],
            test=[x for ex in self.test for x in func(ex)],
        )

    def print_head(self, n=5):
        for name, data in self.get_all():
            print(name)
            for x in data[:n]:
                print('\t', x, sep='')


def load_sentihood(path: Path):
    def _read(file: Path):
        with file.open() as f:
            return json.load(f)

    def process_text(ex):
        text = ex['text'].strip()
        tokens = tokenizer.tokenize(text)
        ex['text'] = tokens

    path = Path(path)
    ds = Dataset(
        _read(path / 'sentihood-train.json'),
        _read(path / 'sentihood-dev.json'),
        _read(path / 'sentihood-test.json'),
    )
    ds.apply(process_text)
    return ds


def main(data_dir: Path):
    ds = load_sentihood(data_dir)

    for name, data in ds.get_all():
        print(name)
        print(
            f"There are {len(data)} {name} examples, "
            f"{sum(len(s['opinions']) for s in data)} opinions, "
            f"{sum(s['opinions'] == [] for s in data)} sentences without opinion"
        )
        print("Aspects:", Counter(op['aspect'] for ex in data for op in ex['opinions']))
        print(
            "Sentiments:",
            Counter(op['sentiment'] for ex in data for op in ex['opinions']),
        )

    aspects = {'general': 0, 'price': 1, 'transit-location': 2, 'safety': 3}
    aspects_token_ids = {
        aspect: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(aspect.replace('-', ' ')))
        for aspect in aspects
    }

    ds.print_head()

    def flatten_aspects(ex):
        # text = [ ('[MASK]' if tok in LOCATIONS else tok) for tok in ex['text'] ]
        ids = tokenizer.convert_tokens_to_ids(ex['text'])
        targets = [loc for loc in LOCATIONS if loc in ex['text']]
        for target in targets:
            target_idx = ex['text'].index(target)
            for aspect in aspects:
                sentiment_or_none = next(
                    (
                        op['sentiment']
                        for op in ex['opinions']
                        if op['target_entity'] == target and op['aspect'] == aspect
                    ),
                    'None',
                )
                yield Example(
                    id=ex['id'],
                    text=ex['text'],
                    segment_ids=ids + [SEP] + aspects_token_ids[aspect],
                    target=target,
                    target_idx=target_idx,
                    aspect=aspect,
                    sentiment=sentiment_or_none,
                )

    processed = ds.map_many(flatten_aspects)

    processed.print_head()

    model = BertModel.from_pretrained(MODEL)
    model.eval()

    for ex in processed.train:
        tensor = torch.tensor([ex.segment_ids])

        predictions, _ = model(tensor)
        print(predictions)
        predicted_index = torch.argmax(predictions[0, ex.target_idx]).item()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)

        print(predicted_index)
        print(predicted_token)
        
        break
