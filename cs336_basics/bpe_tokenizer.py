from collections import defaultdict
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
from multiprocessing import Process, Queue


class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

    def pretokenize(self, chunk: str) -> list[bytes]:
        delimiter = "|".join(
            re.escape(special_token)
            for special_token in sorted(self.special_tokens, key=len, reverse=True)
        )
        chunk_segments = re.split(delimiter, chunk)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pattern = re.compile(PAT)

        pretokens = []
        for chunk_segment in chunk_segments:
            for match in pattern.finditer(chunk_segment):
                pretoken = match.group(0).encode("utf-8")
                pretokens.append(pretoken)

        return pretokens

    def worker(self, chunk: str, q: Queue) -> None:
        pretokens = self.pretokenize(chunk)
        q.put(pretokens)

    def train(self, input_path):
        # 1. Vocabulary initialization
        print("1. Initializing vocabulary...")
        self.vocab = {i: bytes([i]) for i in range(256)}
        for special_token in self.special_tokens:
            self.vocab[len(self.vocab)] = special_token.encode("utf-8")

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
            p = Process(target=self.worker, args=(chunk, q))
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

        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pair_with_max_count = max(
                pair_counts.items(),
                key=lambda x: (
                    x[1],
                    self.vocab[x[0][0]].decode("utf-8", errors="ignore"),
                    self.vocab[x[0][1]].decode("utf-8", errors="ignore"),
                ),
            )[0]
            if pair_counts[pair_with_max_count] == 0:
                break

            a, b = pair_with_max_count
            new_token = self.vocab[a] + self.vocab[b]
            if new_token in self.vocab.values():
                del pair_counts[pair_with_max_count]
                del pair_indices[pair_with_max_count]
                continue

            new_token_id = len(self.vocab)
            self.vocab[new_token_id] = new_token
            self.merges.append((self.vocab[a], self.vocab[b]))

            self.merge(
                pretokens, pair_counts, pair_indices, pair_with_max_count, new_token_id
            )

            # Update old pair counts
            del pair_counts[pair_with_max_count]
            del pair_indices[pair_with_max_count]

        # 4. Return vocab and merges
        print("4. Training complete.")
        return self.vocab, self.merges

    def merge(
        self,
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
                        pair_counts[
                            (pair_with_max_count[1], pair_with_max_count[0])
                        ] -= 1
                    else:  # x, (a, b)
                        pair_counts[
                            (new_pretoken[new_j - 1], pair_with_max_count[0])
                        ] -= 1
                    pair_counts[(new_pretoken[new_j - 1], new_token_id)] += 1
                    pair_indices[(new_pretoken[new_j - 1], new_token_id)].add(i)

                if new_j < len(new_pretoken) - 1:
                    if new_pretoken[new_j + 1] == new_token_id:  # (a, b), (a, b)
                        pair_counts[
                            (pair_with_max_count[1], pair_with_max_count[0])
                        ] -= 1
                    else:  # (a, b), y
                        pair_counts[
                            (pair_with_max_count[1], new_pretoken[new_j + 1])
                        ] -= 1
                    pair_counts[(new_token_id, new_pretoken[new_j + 1])] += 1
                    pair_indices[(new_token_id, new_pretoken[new_j + 1])].add(i)

            pretokens[i] = new_pretoken

    def encode(self, text):
        # Implement the encoding logic using the trained BPE model
        pass

    def decode(self, tokens):
        # Implement the decoding logic to convert tokens back to text
        pass
