import argparse


def main(params):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs", "--batch_size", default=2)

    args = parser.parse_args()

    main(args)
