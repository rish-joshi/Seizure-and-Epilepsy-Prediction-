""" Animated EEG Data for analysis."""

from vedo import *
import numpy as np
import scipy.fft as fft
import time

settings.default_font = "Theemim"
file = np.load("/home/rishabhj/Downloads/x_test.npy")
mycmap = ["darkblue", "magenta", (1, 1, 0)]

x_vector = np.mod(np.arange(500 * 19), 500).reshape(9500, -1)
x_vector = 1 * x_vector / np.max(x_vector)
y_vector = np.zeros(500 * 19)
z_vector = np.zeros(500 * 19)

# Modifying the Z vector
for i in range(19):
	z_vector[500 * i : 500 * (i + 1)] = i / 5
	y_vector[500 * i : 500 * (i + 1)] = file[0][i][:]

y_vector = 3 * y_vector.reshape(9500, -1) / np.max(y_vector)
z_vector = z_vector.reshape(9500, -1) / np.max(z_vector)



final_vector = Points(np.hstack( (x_vector, file[0][:][:].reshape(-1, 1), z_vector) ) )
out = show(final_vector, __doc__, elevation=30)
for i in range(0, file.shape[0], file.shape[0] // 41):
    time.sleep(.20)
    final_vector = final_vector.points(np.hstack( (x_vector, file[i][:][:].reshape(-1, 1), z_vector) ) ).cmap("seismic", x_vector)
    out.render()


out.interactive().close()
