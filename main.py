#/usr/bin/env python3

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument

    arg('--data_dir', default='data/sentihood/')

    args = parser.parse_args()

    from sentihood import main
    main(**vars(args))
