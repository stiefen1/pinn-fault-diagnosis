from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle

from src.excitation.base import ExcitationSignal1D

from typing import Optional, List, Tuple

import math, random

class Zero(ExcitationSignal1D):
    def __init__(
            self
    ):
        super().__init__()

    def __get__(self, t: float) -> float:
        return 0.0

class Step(ExcitationSignal1D):
    def __init__(
            self,
            amplitude: float = 1.0,
            time_delay_sec: float = 0.0,
            y_offset: float = 0.0
    ):
        super().__init__(time_delay_sec=time_delay_sec, y_offset=y_offset)
        self.amplitude = amplitude

    def __get__(self, t: float) -> float:
        return self.amplitude if t >= 0 else 0.0

class Sinus(ExcitationSignal1D):
    def __init__(self, frequency: float = 1.0, amplitude: float = 1.0, time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec, y_offset=y_offset)
        self.frequency = frequency
        self.amplitude = amplitude

    def __get__(self, t: float) -> float:
        return self.amplitude * math.sin(2 * math.pi * self.frequency * t)

class Dirac(ExcitationSignal1D):
    def __init__(self, amplitude: float = 1.0, width: float = 0.01, time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec, y_offset=y_offset)
        self.amplitude = amplitude
        self.width = width

    def __get__(self, t: float) -> float:
        return self.amplitude if -self.width / 2 <= t <= self.width / 2 else 0.0

class PRBS(ExcitationSignal1D):
    def __init__(self, amplitude: float = 1.0, bit_rate: float = 1.0, polynomial: int = 0b1001, seed: int = 1, time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec=time_delay_sec, y_offset=y_offset)
        self.amplitude = amplitude
        self.bit_rate = bit_rate
        self.polynomial = polynomial
        self.register = seed if seed != 0 else 1  # Initial state (avoid 0 which would lock the LFSR)
        assert 1 <= self.register <= 15, f"seed must in [1, 15], got seed={self.register}" 

    def __get__(self, t: float) -> float:
        if t < 0:
            return 0.0
        
        # Calculate which bit period we're in
        bit_index = int(t * self.bit_rate)
        
        # Generate PRBS sequence using LFSR
        register = self.register
        for _ in range(bit_index):
            # Get feedback bit (XOR of tapped bits)
            feedback = 0
            temp_poly = self.polynomial
            temp_reg = register
            while temp_poly:
                if temp_poly & 1:
                    feedback ^= temp_reg & 1
                temp_poly >>= 1
                temp_reg >>= 1
            
            # Shift register and insert feedback
            register = ((register >> 1) | (feedback << 3)) & 0xF
        
        # Return amplitude or -amplitude based on LSB
        return self.amplitude if register & 1 else -self.amplitude

class Pulse(ExcitationSignal1D):
    def __init__(self, amplitude: float = 1.0, width: float = 1.0, time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec=time_delay_sec, y_offset=y_offset)
        self.amplitude = amplitude
        self.width = width

    def __get__(self, t: float) -> float:
        # Rectangular pulse: amplitude during [0, width], 0 elsewhere
        if 0 <= t <= self.width:
            return self.amplitude
        return 0.0

class GaussianNoise(ExcitationSignal1D):
    def __init__(self, amplitude: float = 1.0, seed: Optional[int] = None, time_delay_sec: float = 0.0, y_offset: float = 0.0):
        super().__init__(time_delay_sec=time_delay_sec, y_offset=y_offset)
        self.amplitude = amplitude
        self.reset(seed=seed)
        
    def __get__(self, t: float) -> float:
        # Generate Gaussian noise using Box-Muller transform
        u1 = self._random.random()
        u2 = self._random.random()
        
        # Box-Muller transform
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return self.amplitude * z0
    
    def reset(self, seed: Optional[int] = None) -> None:
        random.seed(seed)
        self._random = random

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    s1 = Sinus(0.1, time_delay_sec=1, y_offset=0.3)
    s2 = Sinus.from_str("Sinus(5, 0.3) + PRBS(0.5, bit_rate=3.0, y_offset=0.5) - GaussianNoise(0.05, seed=42)")
    ax = s1.plot(tf_sec=10.0)
    s2.plot(tf_sec=12.0, ax=ax, sample_rate_hz=100)
    (s1+s2).plot(tf_sec=10.0, sample_rate_hz=500)
    plt.show()