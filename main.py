#!/usr/bin/env python3

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument

    arg('--data_dir', default='data/sentihood/', help="Path to the dataset directory")
    arg('-e', '--epochs', default=1, type=int, help="Training epochs")
    arg(
        '-b',
        '--batch_size',
        default=4,
        type=int,
        help="Batch size for training and evaluation",
    )
    arg(
        '-bs',
        '--balanced_sampler',
        default=True,
        type=bool,
        help="Pick examples uniformly *between classes*",
    )
    arg(
        '-lr',
        '--learning_rate',
        default=1e-4,
        type=float,
        help="Optimizer learning rate",
    )
    arg(
        '-wd',
        '--weight_decay',
        default=0.01,
        type=float,
        help="Optimizer weight decay (L2)",
    )
    arg('--cpu', action='store_true', help="Force CPU even if CUDA is available")
    arg(
        '--debug',
        action='store_true',
        help="Debug options (truncate the datasets for faster debugging)",
    )

    args = parser.parse_args()

    import torch
    from sentihood import main

    args.device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    # main(**vars(args))
    main(args)
