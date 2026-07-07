from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt, numpy as np
from matplotlib.axes import Axes
import re

class ExcitationSignal1D(ABC):
    def __init__(
            self,
            time_delay_sec: float = 0.0,
            y_offset: float = 0.0
    ):
        self.time_delay_sec = time_delay_sec
        self.y_offset = y_offset

    @abstractmethod
    def __get__(self, t_sec: float) -> float:
        pass

    def __call__(self, t_sec: float) -> float:
        return self.__get__(t_sec - self.time_delay_sec) + self.y_offset

    @staticmethod
    def _clean_expression(expr: str) -> str:
        if not expr or not expr.strip():
            raise ValueError("Expression must be a non-empty string")
        return expr.replace(" ", "")

    @staticmethod
    def _fullmatch_expression(clean_expr: str, pattern: str, error_message: str) -> re.Match[str]:
        match = re.fullmatch(pattern, clean_expr)
        if match is None:
            raise ValueError(error_message)
        return match

    @staticmethod
    def _resolve_scalar(token: str, values: Optional[Dict[str, float]] = None) -> float:
        try:
            return float(token)
        except ValueError:
            if not token:
                raise ValueError("Empty scalar token")

            sign = 1.0
            name = token
            if token[0] == "+":
                name = token[1:]
            elif token[0] == "-":
                sign = -1.0
                name = token[1:]

            if not name:
                raise ValueError(f"Invalid scalar token '{token}'")

            values = values or {}
            if name not in values:
                raise ValueError(f"Missing value for symbol '{name}'")

            return sign * float(values[name])

    @classmethod
    def _parse_constructor_kwargs(cls, expr: str, class_name: str) -> Dict[str, str]:
        """Parse rigid constructor form: ClassName(key=value, key2=value2)."""
        clean = cls._clean_expression(expr)
        prefix = f"{class_name}("
        if not clean.startswith(prefix) or not clean.endswith(")"):
            raise ValueError(f"Expected format: {class_name}(key=value,...)" )

        body = clean[len(prefix):-1]
        if body == "":
            return {}

        kwargs: Dict[str, str] = {}
        for item in body.split(","):
            if "=" not in item:
                raise ValueError(f"Invalid argument '{item}' in '{expr}'")
            key, value = item.split("=", 1)
            if key == "" or value == "":
                raise ValueError(f"Invalid key/value '{item}' in '{expr}'")
            if key in kwargs:
                raise ValueError(f"Duplicate argument '{key}' in '{expr}'")
            kwargs[key] = value

        return kwargs
    
    def plot(self, tf_sec: float, *args, t0_sec: float = 0.0, sample_rate_hz: float = 10.0, ax: Optional[Axes] = None, **kwargs) -> Axes:
        n_samples = int((tf_sec - t0_sec) * sample_rate_hz)
        t_vec_sec = np.linspace(t0_sec, tf_sec, n_samples, endpoint=False)
        signal_values = np.array([self(t) for t in t_vec_sec])

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(t_vec_sec, signal_values, *args, **kwargs)
        return ax
    
    def __add__(self, other: "ExcitationSignal1D") -> "SummedSignal1D":
        if isinstance(other, SummedSignal1D):
            signals = [self] + other.signals
        else:
            signals = [self, other]
        return SummedSignal1D(signals)
    
    def __mul__(self, other: "ExcitationSignal1D") -> "MultipliedSignal1D":
        if isinstance(other, MultipliedSignal1D):
            signals = [self] + other.signals
        else:
            signals = [self, other]
        return MultipliedSignal1D(signals)

    def __neg__(self) -> "NegatedSignal1D":
        return NegatedSignal1D(self)

    def __sub__(self, other: "ExcitationSignal1D") -> "SummedSignal1D":
        return self + (-other)

    @classmethod
    def _all_signal_subclasses(cls) -> List[type["ExcitationSignal1D"]]:
        subclasses: List[type["ExcitationSignal1D"]] = []
        for subcls in cls.__subclasses__():
            subclasses.append(subcls)
            subclasses.extend(subcls._all_signal_subclasses())
        return subclasses

    @classmethod
    def from_str(cls, expr: str, values: Optional[Dict[str, Any]] = None) -> "ExcitationSignal1D":
        namespace: Dict[str, object] = {"ExcitationSignal1D": ExcitationSignal1D}
        for subcls in ExcitationSignal1D._all_signal_subclasses():
            namespace[subcls.__name__] = subcls
        namespace[cls.__name__] = cls
        if values is not None:
            namespace.update(values)
        signal = eval(expr, {}, namespace)
        if not isinstance(signal, ExcitationSignal1D):
            raise TypeError(f"Expression must evaluate to ExcitationSignal1D, got {type(signal).__name__}")
        return signal
    
class SummedSignal1D(ExcitationSignal1D):
    def __init__(self, signals: List[ExcitationSignal1D], time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec=time_delay_sec, y_offset=y_offset)
        self.signals = signals

    def __get__(self, t: float) -> float:
        return sum(signal(t) for signal in self.signals)
    
    def __add__(self, other: ExcitationSignal1D) -> "SummedSignal1D":
        if isinstance(other, SummedSignal1D):
            signals = self.signals + other.signals
        else:
            signals = self.signals + [other]
        return SummedSignal1D(signals)
    
    def __repr__(self) -> str:
        return f"Summed({', '.join(str(s) for s in self.signals)})"

class NegatedSignal1D(ExcitationSignal1D):
    def __init__(self, signal: ExcitationSignal1D, time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec=time_delay_sec, y_offset=y_offset)
        self.signal = signal

    def __get__(self, t: float) -> float:
        return -self.signal(t)

    def __repr__(self) -> str:
        return f"Negated({self.signal})"
    
class MultipliedSignal1D(ExcitationSignal1D):
    def __init__(self, signals: List[ExcitationSignal1D], time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec=time_delay_sec, y_offset=y_offset)
        self.signals = signals

    def __get__(self, t: float) -> float:
        result = 1.0
        for signal in self.signals:
            result *= signal(t)
        return result
    
    def __mul__(self, other: ExcitationSignal1D) -> "MultipliedSignal1D":
        if isinstance(other, MultipliedSignal1D):
            signals = self.signals + other.signals
        else:
            signals = self.signals + [other]
        return MultipliedSignal1D(signals)
    
    def __repr__(self) -> str:
        return f"Multiplied({', '.join(str(s) for s in self.signals)})"