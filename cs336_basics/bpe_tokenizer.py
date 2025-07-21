from typing import BinaryIO, Iterable, Iterator
from collections import defaultdict
import regex as re
import os
from multiprocessing import Process, Queue


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(
    text: str, special_tokens: list[str], drop_special_tokens: bool = True
) -> list[bytes]:
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    if not special_tokens_sorted:
        segments = [text]
    else:
        delimiter = "|".join(
            re.escape(special_token) for special_token in special_tokens_sorted
        )
        segments = re.split("(" + delimiter + ")", text)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)

    pretokens = []
    for segment in segments:
        if segment in special_tokens:
            if not drop_special_tokens:  # Keep special tokens, otherwise ignore
                pretokens.append([segment.encode("utf-8")])
        else:
            pretoken = []
            for match in pattern.finditer(segment):
                token = match.group(0).encode("utf-8")
                pretoken.append(token)
            pretokens.append(pretoken)
    return [token for part_tokens in pretokens for token in part_tokens]


def worker(chunk: str, special_tokens: list[str], q: Queue) -> None:
    pretokens = pretokenize(chunk, special_tokens)
    q.put(pretokens)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Vocabulary initialization
    print("1. Initializing vocabulary...")
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []

    # 2. Pre-tokenization
    print("2. Pre-tokenizing input...")
    chunk_list = []
    num_processes = os.cpu_count() or 1
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8")
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    processes = []
    q = Queue()
    for chunk in chunk_list:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        processes.append(p)
        p.start()

    pretokens_list = [q.get() for _ in processes]

    for p in processes:
        p.join()

    pretokens = [list(token) for tokens in pretokens_list for token in tokens]

    # 3. Merge
    print("3. Merging tokens...")
    pair_counts = defaultdict(int)
    pair_indices = defaultdict(set)
    for i, pretoken in enumerate(pretokens):
        for j in range(len(pretoken) - 1):
            pair = (pretoken[j], pretoken[j + 1])
            pair_counts[pair] += 1
            pair_indices[pair].add(i)

    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        pair_with_max_count = max(
            pair_counts.items(),
            key=lambda x: (
                x[1],
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore"),
            ),
        )[0]
        if pair_counts[pair_with_max_count] == 0:
            break

        a, b = pair_with_max_count
        new_token = vocab[a] + vocab[b]
        if new_token in vocab.values():
            del pair_counts[pair_with_max_count]
            del pair_indices[pair_with_max_count]
            continue

        new_token_id = len(vocab)
        vocab[new_token_id] = new_token
        merges.append((vocab[a], vocab[b]))

        merge(pretokens, pair_counts, pair_indices, pair_with_max_count, new_token_id)

        # Update old pair counts
        del pair_counts[pair_with_max_count]
        del pair_indices[pair_with_max_count]

    # 4. Return vocab and merges
    print("4. Training complete.")
    return vocab, merges


def merge(
    pretokens: list[list[int]],
    pair_counts: dict[tuple[int, int], int],
    pair_indices: dict[tuple[int, int], set[int]],
    pair_with_max_count: tuple[int, int],
    new_token_id: int,
) -> None:
    for i in pair_indices[pair_with_max_count]:
        j = 0
        new_pretoken = []
        new_j = 0
        new_token_indices = []
        while j < len(pretokens[i]):
            if (
                j < len(pretokens[i]) - 1
                and (pretokens[i][j], pretokens[i][j + 1]) == pair_with_max_count
            ):
                new_token_indices.append(new_j)
                new_pretoken.append(new_token_id)
                j += 2
            else:
                new_pretoken.append(pretokens[i][j])
                j += 1
            new_j += 1

        for new_j in new_token_indices:
            if new_j > 0:
                if new_pretoken[new_j - 1] == new_token_id:  # (a, b), (a, b)
                    pair_counts[(pair_with_max_count[1], pair_with_max_count[0])] -= 1
                else:  # x, (a, b)
                    pair_counts[(new_pretoken[new_j - 1], pair_with_max_count[0])] -= 1
                pair_counts[(new_pretoken[new_j - 1], new_token_id)] += 1
                pair_indices[(new_pretoken[new_j - 1], new_token_id)].add(i)

            if new_j < len(new_pretoken) - 1:
                if new_pretoken[new_j + 1] == new_token_id:  # (a, b), (a, b)
                    pair_counts[(pair_with_max_count[1], pair_with_max_count[0])] -= 1
                else:  # (a, b), y
                    pair_counts[(pair_with_max_count[1], new_pretoken[new_j + 1])] -= 1
                pair_counts[(new_token_id, new_pretoken[new_j + 1])] += 1
                pair_indices[(new_token_id, new_pretoken[new_j + 1])].add(i)

        pretokens[i] = new_pretoken


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens=None,
    ):
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens_encoded = [
            token.encode("utf-8") for token in self.special_tokens
        ]
        for token in self.special_tokens:
            encoded_token = token.encode("utf-8")
            if encoded_token not in self.vocab.values():
                self.vocab[len(self.vocab)] = encoded_token

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json

        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
            vocab = {int(k): v.encode("utf-8") for k, v in vocab.items()}

        with open(merges_filepath, "r") as f:
            merges = [tuple(line.strip().split()) for line in f]
            merges = [(vocab[int(a)], vocab[int(b)]) for a, b in merges]

        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        # 1. Pre-tokenization
        byted_pretokens = pretokenize(
            text, self.special_tokens, drop_special_tokens=False
        )
        pretokens = []

        # 2. Turn pretokens into list[list[int]]
        for byted_pretoken in byted_pretokens:
            if byted_pretoken in self.special_tokens_encoded:
                pretokens.append([self.vocab_reversed[byted_pretoken]])
            else:
                pretokens.append(
                    [self.vocab_reversed[bytes([b])] for b in byted_pretoken]
                )

        # 3. Encoding
        for merge in self.merges:
            for i, pretoken in enumerate(pretokens):
                new_pretoken = []
                j = 0
                while j < len(pretoken):
                    if (
                        j < len(pretoken) - 1
                        and (self.vocab[pretoken[j]], self.vocab[pretoken[j + 1]])
                        == merge
                    ):
                        new_pretoken.append(self.vocab_reversed[merge[0] + merge[1]])
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1

                pretokens[i] = new_pretoken

        # 4. Flatten the pretokens
        return [token for pretoken in pretokens for token in pretoken]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids: list[int]) -> str:
        tokens = bytes()
        replacement_char = "\ufffd"

        for id in ids:
            if id < len(self.vocab):
                token = self.vocab[id]
            else:
                token = bytes(replacement_char, encoding="utf-8")
            tokens += token

        return tokens.decode("utf-8", errors="replace")
