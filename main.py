#/usr/bin/env python3

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument

    arg('--data_dir', default='data/sentihood/', help="Path to the dataset directory")
    arg('-b', '--batch_size', default=4, help="Batch size for training and evaluation")
    arg('-lr', '--learning_rate', default=1e-4, help="Optimizer learning rate")
    arg('-wd', '--weight_decay', default=0.01, help="Optimizer weight decay (L2)")

    args = parser.parse_args()

    from sentihood import main
    # main(**vars(args))
    main(args)
