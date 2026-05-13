from dataclasses import dataclass, field
import numpy as np

@dataclass
class RunningStats:
    """
    Welford streaming mean/var per feature dimension.
    D = bins*bins after pooling.
    """
    D: int
    n: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    M2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __post_init__(self):
        if self.mean.size == 0:
            self.mean = np.zeros(self.D, dtype=np.float32)
            self.M2 = np.zeros(self.D, dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.full_like(self.mean, 1e-6)
        return self.M2 / (self.n - 1)
        
    def update_batch(self, x_batch: np.ndarray) -> None:
        """
        Updates statistics with a batch of vectors [k, D].
        Mathematically equivalent to calling update() k times, but much faster.
        """
        k = x_batch.shape[0]
        if k == 0:
            return

        batch_mean = np.mean(x_batch, axis=0)
        batch_m2 = np.sum((x_batch - batch_mean) ** 2, axis=0)

        delta = batch_mean - self.mean
        new_n = self.n + k

        # Welford's parallel merge formula
        self.M2 += batch_m2 + (delta ** 2) * (self.n * k / new_n)
        self.mean += delta * (k / new_n)
        self.n = new_n
