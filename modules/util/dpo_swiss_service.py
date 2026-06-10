"""Swiss-tournament scheduling for the DPO Pair Tool's comparison mode.

Replaces unbounded Elo voting with a fixed number of rounds: every image is
paired once per round against a score-adjacent opponent (no rematches while
avoidable), so the total comparison count is known up front. Elo ratings are
still updated per vote, but only as a tiebreaker within equal scores.
"""

import math
import random
from dataclasses import dataclass, field

ELO_K_FACTOR = 32.0
ELO_INITIAL = 1500.0


def swiss_round_count(n: int) -> int:
    """ceil(log2(n)) rounds separates a clear winner; clamped to [2, 5]."""
    return max(2, min(5, math.ceil(math.log2(max(n, 2)))))


def outermost_pairs(order: list[str], k: int) -> list[tuple[str, str]]:
    """First k (chosen, rejected) pairs walking inward from both ends of a
    best-to-worst ranking: rank 1 vs rank n, rank 2 vs rank n-1, and so on.
    Outermost-first maximizes the preference gap of every exported pair."""
    k = max(0, min(k, len(order) // 2))
    return [(order[i], order[-1 - i]) for i in range(k)]


@dataclass
class SwissPlayer:
    image: str
    score: float = 0.0  # win 1.0, tie 0.5, bye 1.0
    elo: float = ELO_INITIAL
    opponents: set[str] = field(default_factory=set)
    byes: int = 0


class SwissTournament:
    def __init__(
            self,
            images: list[str],
            rounds: int | None = None,
            rng: random.Random | None = None,
    ):
        if len(images) < 2:
            raise ValueError("A tournament needs at least two images.")
        self._players = {image: SwissPlayer(image) for image in images}
        self._rng = rng if rng is not None else random.Random()
        self._total_rounds = rounds if rounds is not None else swiss_round_count(len(images))
        self._current_round = 0
        self._queue: list[tuple[str, str]] = []
        self._matches_played = 0

    @property
    def total_rounds(self) -> int:
        return self._total_rounds

    @property
    def current_round(self) -> int:
        return max(1, self._current_round)

    def matches_per_round(self) -> int:
        return len(self._players) // 2

    def matches_total(self) -> int:
        return self._total_rounds * self.matches_per_round()

    def matches_played(self) -> int:
        return self._matches_played

    def is_finished(self) -> bool:
        return not self._queue and self._current_round >= self._total_rounds

    def next_match(self) -> tuple[str, str] | None:
        if not self._queue:
            if self._current_round >= self._total_rounds:
                return None
            self._pair_round()
        return self._queue[0] if self._queue else None

    def report(self, a: str, b: str, result: str):
        """result: 'a', 'b' or 'tie'. Must be the pair returned by next_match."""
        if not self._queue or self._queue[0] != (a, b):
            raise ValueError("report() must resolve the match returned by next_match().")
        self._queue.pop(0)
        self._matches_played += 1

        player_a = self._players[a]
        player_b = self._players[b]
        player_a.opponents.add(b)
        player_b.opponents.add(a)

        score_a = {"a": 1.0, "b": 0.0, "tie": 0.5}[result]
        player_a.score += score_a
        player_b.score += 1.0 - score_a

        expected_a = 1.0 / (1.0 + 10.0 ** ((player_b.elo - player_a.elo) / 400.0))
        player_a.elo += ELO_K_FACTOR * (score_a - expected_a)
        player_b.elo += ELO_K_FACTOR * ((1.0 - score_a) - (1.0 - expected_a))

    def player(self, image: str) -> SwissPlayer:
        return self._players[image]

    def standings(self) -> list[str]:
        return [
            player.image
            for player in sorted(self._players.values(), key=lambda p: (-p.score, -p.elo, p.image))
        ]

    def _pair_round(self):
        self._current_round += 1

        order = self.standings()
        if self._current_round == 1:
            self._rng.shuffle(order)

        unpaired = list(order)

        if len(unpaired) % 2 == 1:
            # Bye to the lowest-ranked player with the fewest byes; a bye
            # scores a free win so sitting out is never a penalty.
            bye_player = min(reversed(unpaired), key=lambda image: self._players[image].byes)
            unpaired.remove(bye_player)
            self._players[bye_player].score += 1.0
            self._players[bye_player].byes += 1

        while unpaired:
            top = unpaired.pop(0)
            fresh = [image for image in unpaired if image not in self._players[top].opponents]
            if fresh:
                opponent = fresh[0]
            else:
                # Only previous opponents remain: allow the closest rematch.
                opponent = min(
                    unpaired,
                    key=lambda image: (
                        abs(self._players[image].score - self._players[top].score),
                        abs(self._players[image].elo - self._players[top].elo),
                    ),
                )
            unpaired.remove(opponent)
            self._queue.append((top, opponent))
