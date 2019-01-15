import json
from collections import Counter
from functools import partial
from pathlib import Path
from typing import List, TypeVar

import attr
import torch
from pytorch_pretrained_bert import (
    BertAdam,
    BertForMaskedLM,
    BertForTokenClassification,
)
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from tokenization import BertTokenizer

T = TypeVar('T')

ARGS = None
MODEL = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(
    MODEL,
    do_lower_case='uncased' in MODEL,
    never_split='[UNK] [SEP] [PAD] [CLS] [MASK] LOCATION1 LOCATION2'.split(),
)

LOCATIONS = ['LOCATION1', 'LOCATION2']
for loc in LOCATIONS:
    tokenizer.vocab[loc] = tokenizer.vocab['[MASK]']

CLS = tokenizer.vocab['[CLS]']
SEP = tokenizer.vocab['[SEP]']

writer = SummaryWriter()
scorers = {
    'accuracy': metrics.accuracy_score,
    'f1_micro': partial(metrics.f1_score, average='micro'),
    'f1_macro': partial(metrics.f1_score, average='macro'),
    'f1_weighted': partial(metrics.f1_score, average='weighted'),
}


@attr.s(auto_attribs=True, slots=True)
class Example:
    """Represent a training or test example"""

    id: int
    text: str
    token_ids: List[int]  # CLS + text + SEP + aspect + SEP
    target: str
    target_idx: int  # in token_ids
    aspect: str
    sentiment: str
    label: int  # sentiment as an int


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


def segment_ids_from_token_ids(token_ids):
    """We want all 0s before, and including, the first SEP and 1s after that if there are remaining tokens"""
    first = token_ids.index(SEP)
    return [int(i > first) for i in range(len(token_ids))]


def pad(arr: List[T], maxlen: int, value: T):
    """len(arr) must be <= maxlen"""
    return arr + [value for _ in range(maxlen - len(arr))]


def create_batch(examples: List[Example]):
    maxlen = max(len(ex.token_ids) for ex in examples)
    tokens = [pad(ex.token_ids, maxlen, 0) for ex in examples]
    segments = [pad(segment_ids_from_token_ids(toks), maxlen, 1) for toks in tokens]
    mask = [pad([1 for _ in tok], maxlen, 0) for tok in tokens]
    # -100 == ignore_index of the cross entropy loss
    labels = [
        [-100 if i != ex.target_idx else ex.label for i in range(maxlen)]
        for ex in examples
    ]
    return (
        torch.tensor(tokens, device=ARGS.device),
        torch.tensor(segments, device=ARGS.device),
        torch.tensor(mask, device=ARGS.device),
        torch.tensor(labels, device=ARGS.device),
    )


def train(model, train_data, eval_data, epochs):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': ARGS.weight_decay,  # 0.01,
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=ARGS.learning_rate,
        warmup=0.1,
        t_total=len(train_data),
    )

    model.train()

    for epoch in trange(epochs, desc="Train epoch"):
        for step, batch in enumerate(tqdm(train_data, desc="Iteration")):
            loss = model(*batch)
            tqdm.write(f"loss={loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('train/loss', loss.item(), step)

        writer.add_graph('bert', model, batch[-1])

        eval(model, eval_data)


def eval(model, data_loader):
    model.eval()
    # eval_metrics = Metrics()
    all_labels = []
    all_preds = []
    # all_probs = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Eval")):
            assert len(batch) == 4, "We should have labels here"
            labels = batch[3]
            targets = (
                labels != -100
            )  # the ignored index in the loss (= we ignore the tokens that not the target)

            logits = model(*batch[:3])
            # probs = F.softmax(logits, dim=1)[:,1]  # TODO check this
            predictions = logits.argmax(dim=-1)[targets].tolist()
            # all_probs += probs.tolist()
            labels = labels[targets].tolist()

            all_preds += predictions
            all_labels += labels

    # writer.add_pr_curve('eval', labels=all_labels, predictions=all_probs)
    tqdm.write(f"labels={' '.join(map(str, all_labels))}")
    tqdm.write(f"preds ={' '.join(map(str, all_preds))}")
    # writer.add_scalar('eval/acc', metrics.accuracy_score(all_labels, all_preds))
    # writer.add_scalar('eval/f1 micro', metrics.f1_score(all_labels, all_preds, ))
    for name, scorer in scorers.items():
        writer.add_scalar(f'eval/{name}', scorer(all_labels, all_preds))
    writer.add_text(
        'eval/classification_report',
        metrics.classification_report(
            all_labels,
            all_preds,
            labels=[0, 1, 2],
            target_names='None Positive Negative'.split(),
        ),
    )


def main(args):
    global ARGS
    ARGS = args
    ds = load_sentihood(ARGS.data_dir)

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
        aspect: tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(aspect.replace('-', ' '))
        )
        for aspect in aspects
    }
    sentiments = {'None': 0, 'Positive': 1, 'Negative': 2}

    ds.print_head()

    def flatten_aspects(ex):
        # text = [ ('[MASK]' if tok in LOCATIONS else tok) for tok in ex['text'] ]
        ids = tokenizer.convert_tokens_to_ids(ex['text'])
        targets = [loc for loc in LOCATIONS if loc in ex['text']]
        for i, target in enumerate(targets):
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
                    token_ids=[CLS] + ids + [SEP] + aspects_token_ids[aspect] + [SEP],
                    target=target,
                    target_idx=1 + target_idx,  # 1 offset for CLS
                    aspect=aspect,
                    sentiment=sentiment_or_none,
                    label=sentiments[sentiment_or_none],
                )

    processed = ds.map_many(flatten_aspects)
    if ARGS.debug:
        processed.train = processed.train[: 2 * ARGS.batch_size]
        processed.dev = processed.dev[: 2 * ARGS.batch_size]
        processed.test = processed.test[: 2 * ARGS.batch_size]

    processed.print_head()

    writer.add_text('params', f"model={MODEL} params={str(ARGS)}")

    # 3 labels for None/neutral, Positive, Negative
    model = BertForTokenClassification.from_pretrained(MODEL, num_labels=3)
    model.to(ARGS.device)

    if ARGS.balanced_sampler:
        class_counts = Counter(ex.sentiment for ex in processed.train)
        class_min = class_counts.most_common()[-1][1]
        writer.add_text(
            'info/balanced_sampler_weights',
            str(
                {
                    sentiment: class_min / count
                    for sentiment, count in class_counts.items()
                }
            ),
        )
        weights = [
            len(processed.train) / class_counts[ex.sentiment] for ex in processed.train
        ]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(processed.train)
        )
    else:
        sampler = None

    train_loader = DataLoader(
        processed.train,
        batch_size=ARGS.batch_size,
        shuffle=not ARGS.balanced_sampler,
        sampler=sampler,
        collate_fn=create_batch,
    )
    eval_loader = DataLoader(
        processed.dev, batch_size=ARGS.batch_size, collate_fn=create_batch
    )

    train(model, train_loader, eval_loader, epochs=ARGS.epochs)
