import random
from typing import List, Set, Tuple, Optional
from pathlib import Path


def load_docs(corpus_path: Path,
              shuffle_docs: Optional[bool] = False,
              shuffle_sentences: Optional[bool] = False,
              test_doc_ids: Optional[List[int]] = None,
              num_test_docs: Optional[int] = 100,
              shuffle_seed: Optional[int] = 20,
              split_seed: Optional[int] = 3
              ) -> Tuple[List[str], List[str]]:

    text_in_file = corpus_path.read_text()
    docs = text_in_file.split('\n')
    num_docs = len(docs)
    print(f'Loaded {num_docs} documents from {corpus_path}')

    # shuffle at sentence-level (as opposed to document-level)
    # this remove clustering of same-age utterances within documents
    if shuffle_sentences:
        random.seed(shuffle_seed)
        print('WARNING: Shuffling sentences')
        tokens = text_in_file.replace('\n', ' ').split()
        sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
        random.shuffle(sentences)
        # assign sentences to documents + convert back to strings
        num_s_in_doc = len(sentences) // num_docs
        print(f'Found {len(sentences)} sentences')
        print(f'Assigning {num_s_in_doc} sentences to each new document')
        docs = [' '.join([w for s in s_chunk for w in s]) for s_chunk in split(sentences, num_s_in_doc)]
        print(f'After shuffling, number of docs={len(docs)}')

    # split train/test
    print('Splitting docs into train and test...')
    if test_doc_ids is None:
        num_test_doc_ids = num_docs - num_test_docs
        random.seed(split_seed)
        test_doc_ids = random.sample(range(num_test_doc_ids), num_test_docs)
    else:
        print('WARNING: Using custom test-doc-ids.')

    test_docs = []
    for test_line_id in test_doc_ids:
        test_doc = docs.pop(test_line_id)  # removes line and returns removed line
        test_docs.append(test_doc)

    # shuffle after train/test split
    if shuffle_docs:
        print('Shuffling documents')
        random.seed(shuffle_seed)
        random.shuffle(docs)

    print(f'Collected {len(docs):,} train docs')
    print(f'Collected {len(test_docs):,} test docs')

    return docs, test_docs


def split_into_sentences(tokens: List[str],
                         punctuation: Set[str],
                         ) -> List[List[str]]:
    assert isinstance(punctuation, set)

    res = [[]]
    for n, w in enumerate(tokens):
        res[-1].append(w)
        if w in punctuation and n < len(tokens) - 1:  # prevent appending empty list at end
            res.append([])
    return res


def split(l, split_size):
    for i in range(0, len(l), split_size):
        yield l[i:i + split_size]