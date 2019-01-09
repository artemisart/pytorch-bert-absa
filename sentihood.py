from collections import Counter
import json
from pathlib import Path

import attr
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

MODEL = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(MODEL)


@attr.s(auto_attribs=True, slots=True)
class Dataset:
    train: list
    dev: list
    test: list

    def get_all(self):
        for field in self.__slots__:
            yield field, getattr(self, field)

    @classmethod
    def load(cls, path: Path):
        def _read(file: Path):
            with file.open() as f:
                return json.load(f)

        path = Path(path)
        return cls(
            _read(path / 'sentihood-train.json'),
            _read(path / 'sentihood-dev.json'),
            _read(path / 'sentihood-test.json'),
        )


def parse_sentihood(in_file):
    with open(in_file) as file:
        sentences = json.load(file)
    for s in sentences:
        s['text'] = s['text'].strip()


def main(args):
    ds = Dataset.load(args.data_dir)
    # print(ds)
    print(ds.__slots__)

    for name, data in ds.get_all():
        print(f"""There are {len(data)} {name} examples,
            {sum(len(s['opinions']) for s in data)} opinions,
            {sum(s['opinions'] == [] for s in data)} sentences without opinion""")


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument

    arg('--data_dir', default='data/sentihood/')

    args = parser.parse_args()
    main(args)
