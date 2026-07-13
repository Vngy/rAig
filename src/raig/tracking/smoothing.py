import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Casiez et al. 2012 — low-latency low-jitter smoothing."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None or self._t_prev is None or t <= self._t_prev:
            self._x_prev, self._t_prev = x, t
            return x
        dt = t - self._t_prev
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


class FilterBank:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, OneEuroFilter] = {}

    def apply(self, values: dict[str, float], t: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, v in values.items():
            f = self._filters.get(name)
            if f is None:
                f = self._filters[name] = OneEuroFilter(self.min_cutoff, self.beta)
            out[name] = f(v, t)
        return out
