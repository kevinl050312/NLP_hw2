#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""
# JHU NLP HW2
# Name: Kevin Li
# Email: kli40@jhu.edu
# Term: Fall 2022

from __future__ import annotations
import argparse
import logging
import re
import numpy as np
from pathlib import Path
from integerize import Integerizer  # look at integerize.py for more info

# For type annotations, which enable you to check correctness of your code:
from typing import List, Optional, Type, Any

try:
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import torch as th
    import torch.nn as nn
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise

log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.


# Logging is in general a good practice to check the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# - It prints to standard error (stderr), not standard output (stdout) by
#   default. This means it won't interfere with the real output of your
#   program. 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
#
# In `parse_args`, we provided two command line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'. 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args


class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.
    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words(bagpipe)
    """

    def __init__(self) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
        # FINISH THIS FUNCTION
        self.vocab = None           # an Integerizer object based on a list of the words in the lexicon
        self.nwords = None          # number of words in the file
        self.matrix = None        # The embedding matrix in tensor form
        # self.from_file(lexicon_file)

        # Store your stuff! Both the word-index mapping and the embedding matrix.
    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        embedlist = []
        wordlist = []

        with open(file) as f:
            first_line = next(f)    # Peel off the special first line.
            for line in f:          # All of the other lines are regular.
                line = line.strip('\n')
                splitline = re.split('\t', line)
                wordlist.append(splitline[0])
                embedlist.append([float(i) for i in splitline[1:]])

        lexicon = Lexicon() # Maybe put args here. Maybe follow Builder pattern.
        lexicon.matrix = th.tensor(embedlist)
        lexicon.vocab = Integerizer(wordlist)  # does this need to be a tensor as well?
        lexicon.nwords = len(lexicon.vocab)

        return lexicon

    def find_similar_words(
            self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ):
        """Find most similar words, in terms of embeddings, to a query."""
        # The star above forces you to use `plus` and `minus` as
        # named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.

        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")

        # Be sure that you use fast, batched computations
        # instead of looping over the rows. If you use a loop or a comprehension
        # in this function, you've probably made a mistake.
        cos1 = nn.CosineSimilarity(dim=1)           # initialize cosine similarity function, compares rows
        word_idx = self.vocab.index(word)           # the index of the given word in the vocab
        word_embed = self.matrix[word_idx]          # the vector embedding of the given word

        # If we are given minus and plus, get the embeddings for those words and add/subtract from first word
        if plus and minus:
            plus_idx = self.vocab.index(plus)
            minus_idx = self.vocab.index(minus)
            remove_idxs = [word_idx, plus_idx, minus_idx]
            plus_embed = self.matrix[plus_idx]
            minus_embed = self.matrix[minus_idx]

            word_embed = th.add(th.subtract(word_embed, minus_embed), plus_embed)

        # Approach: map operation to whole tensor, then call min on generated cosine values
        # define a matrix with every row the given word's embedding and call cos on that and full matrix
        # need to define vector of the given word then apply the cosine similarity formula to every other row of the tensor
        oneword_matrix = word_embed.repeat(self.nwords, 1)
        sim_vals = cos1(oneword_matrix, self.matrix)            # This is a tensor already

        if plus and minus:
            # Case with plus/minus
            # Get indices of top 13 similarity scores, remove indices of the three words, get top 10 of remaining
            [top_vals, top_idxs] = th.topk(sim_vals, 13)
            top_idxs = top_idxs.tolist()
            for i in top_idxs:
                for j in remove_idxs:
                    if i==j:
                        top_idxs.remove(i)
            top_idxs = top_idxs[:10]
        else:
            # Case without plus/minus
            # Get indices of top 11 similarity scores, exclude 0th since it will be the input word
            [top_vals, top_idxs] = th.topk(sim_vals, 11)
            top_idxs = top_idxs.tolist()[1:]

        sim_words = [self.vocab[i] for i in top_idxs]

        return sim_words


def format_for_printing(word_list: List[str]) -> str:
    words_formatted = ''
    for i in word_list:
        words_formatted += i + ' '
    # We don't print out the list as-is; the handout
    # asks that you display it in a particular way.
    # FINISH THIS FUNCTION
    return words_formatted


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        word=args.word, plus=args.plus, minus=args.minus
    )
    print(format_for_printing(similar_words))


if __name__ == "__main__":
    main()
