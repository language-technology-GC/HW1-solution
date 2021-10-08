#!/usr/bin/env python
"""Computes word2vec embeddings."""


import argparse
import logging

from gensim.models import word2vec  # type: ignore


def main(args: argparse.Namespace) -> None:
    # Trains w2v model.
    sentences = word2vec.LineSentence(args.tok_path)
    w2v = word2vec.Word2Vec(
        sentences,
        epochs=args.epochs,
        min_count=args.min_count,
        vector_size=args.vector_size,
        window=args.window,
        workers=4,
    )
    w2v.wv.save(args.model_path)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs (default: %(default)s)",
    )
    parser.add_argument("--min_count", default=5, help="min count")
    parser.add_argument(
        "--vector_size",
        default=100,
        help="embedding size (default: %(default)s)",
    )
    parser.add_argument(
        "--tok_path", required=True, help="path to input tokenized text file"
    )
    parser.add_argument(
        "--window",
        default=5,
        help="symmetric window size (default: %(default)s)",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="path to output word2vec model file",
    )
    main(parser.parse_args())
