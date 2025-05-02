import numpy as np
import scipy
from scipy.io.wavfile import read, write


class TripleNestedGardnerAllpass:
    def __init__(
        self, ig=0.6, sr=44100, delay_samples=[501, 707, 911, 1581], dry_wet=0.5
    ):
        self.ig = ig
        self.sr = sr
        self.M1, self.M2, self.M3, self.M4 = delay_samples

        # Initialize delay buffers with proper sizes
        self.adel1 = np.zeros(self.M1)
        self.adel2 = np.zeros(self.M2)
        self.adel3 = np.zeros(self.M3)
        self.adel4 = np.zeros(self.M4)

        # Initialize indices for circular buffers
        self.idx1 = 0
        self.idx2 = 0
        self.idx3 = 0
        self.idx4 = 0

        # Initialize feedback variables
        self.aout1 = 0
        self.aout2 = 0
        self.aout3 = 0
        self.aout4 = 0

        self.dry_wet = dry_wet

    def process(self, x):
        y = np.zeros_like(x)
        for n in range(len(x)):
            a1 = x[n]

            # Outer delay and feedback
            adel4_out = self.adel4[self.idx4]
            self.adel4[self.idx4] = a1 + self.ig * self.aout4
            self.idx4 = (self.idx4 + 1) % self.M4

            # Feed forward and delay for adel1
            self.aout1 = self.adel1[self.idx1] - self.ig * adel4_out
            self.adel1[self.idx1] = adel4_out + self.ig * self.aout1
            self.idx1 = (self.idx1 + 1) % self.M1

            # Feed forward and delay for adel2
            self.aout2 = self.adel2[self.idx2] - self.ig * self.aout1
            self.adel2[self.idx2] = self.aout1 + self.ig * self.aout2
            self.idx2 = (self.idx2 + 1) % self.M2

            # Feed forward and delay for adel3
            self.aout3 = self.adel3[self.idx3] - self.ig * self.aout2
            self.adel3[self.idx3] = self.aout2 + self.ig * self.aout3
            self.idx3 = (self.idx3 + 1) % self.M3

            # Outer feed forward
            self.aout4 = self.aout3 - self.ig * a1

            # Output (mix dry and wet, 50/50)
            y[n] = (1 - self.dry_wet) * x[n] + (self.dry_wet) * self.aout4
        return y


if __name__ == "__main__":
    path_to_solo_instr = "flute.wav"
    sr, flute_data = read(path_to_solo_instr)
    
    room_settings = [200, 500, 700, 1200]
    medium_hall_settings = [500, 700, 900, 1500]
    cathedral_settings = [500, 1000, 1500, 2000]
    
    reverb_effect = TripleNestedGardnerAllpass(
        sr=sr,
        delay_samples=cathedral_settings,
        dry_wet=0.35,
    )

    y = reverb_effect.process(flute_data)

    write("flute_out_cathedral.wav", rate=sr, data=y)
