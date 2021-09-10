#!/usr/bin/env python
"""Computes correlation for word2vec similarity measures."""

import argparse
import logging
import csv

from gensim.models import word2vec  # type: ignore
from scipy import stats  # type: ignore


def main(args: argparse.Namespace) -> None:
    wv = word2vec.KeyedVectors.load(args.model_path)
    # Computes cosine similarities for targeted pairs.
    with open(args.ws353_path, "r") as source:
        reader = csv.reader(source, delimiter="\t")
        covered = 0
        seen = 0
        word2vec_sim = []
        human_sim = []
        for (x, y, sim) in reader:
            try:
                word2vec_sim.append(wv.similarity(x, y))
                human_sim.append(float(sim))
                covered += 1
            except KeyError:
                pass
            seen += 1
        rho = stats.spearmanr(word2vec_sim, human_sim).correlation
        coverage = covered / seen
        logging.info("word2vec:\t% .4f (coverage: %.4f)", rho, coverage)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ws353_path",
        required=True,
        help="path to ws353 TSV file",
    )
    parser.add_argument(
        "--model_path", required=True, help="path to input word2vec model file"
    )

    main(parser.parse_args())
