import matplotlib.pyplot as plt
import numpy as np

import array_index
import aoa
import aoa_1d
import config

def cal(A: list,channel):
    I = A[0::2]
    Q = A[1::2]
    S = [complex(I[i], -Q[i]) for i in range(len(I))]
    d = np.diff(np.angle(S))
    d = np.mod(d, 2 * np.pi)
    dphase = np.mean(d[0:7])

    phaseS = np.angle(S)
    for i in range(8):
        phaseS[i] = np.mod(phaseS[i] - dphase * i, 2 * np.pi)
    for i in range(8, len(phaseS)):
        phaseS[i] = np.mod(phaseS[i] - dphase * 7 - dphase * (i - 7) * 2, 2 * np.pi)
    phaseS = phaseS[7:]
    temp = phaseS[33:66] - phaseS[0:33]
    tempd = np.median(temp) / 33
    print(dphase+tempd/2)
    for i in range(len(phaseS)):
        phaseS[i] = phaseS[i] - tempd * i
    if config.polar_mode:
        phaseS = phaseS.reshape((1, -1))#silicon antenna not need reindex
        music_worker_2D = aoa.Music()
        pola = 1  # 0 V or 1 H
        phase = -phaseS[0, 1 + pola:32 + pola:2]
        phase = phase.reshape((16, 1))
        phase2 = -phaseS[0, 34 + pola:65 + pola:2]
        phase2 = phase2.reshape((16, 1))
        phase = np.hstack((phase, phase2))
        signal = np.exp(1j * phase)
        psp, azimuth, elevation, intensity = music_worker_2D.music(signal, channel)
        pola = 0  # 0 V or 1 H
        phase = -phaseS[0, 1 + pola:32 + pola:2]
        phase = phase.reshape((16, 1))
        phase2 = -phaseS[0, 34 + pola:65 + pola:2]
        phase2 = phase2.reshape((16, 1))
        phase = np.hstack((phase, phase2))
        signal = np.exp(1j * phase)
        psp2, azimuth2, elevation2, intensity2 = music_worker_2D.music(signal, channel)
        # pp = np.append(psp,psp2)
        # if intensity2 > intensity:
        #     psp, azimuth, elevation, intensity = psp2, azimuth2, elevation2, intensity2
    else:
        phaseS = array_index.rindex_array(phaseS)
        initial_phase=np.array([-102.4363234,-99.8093776,-97.89510349,-96.79578985,-98.01015639,-96.06824975,-93.86151304,-90.29052919,-89.0186968,-89.50369698,-89.75412113,-92.48015795,-91.20332602,-94.52531665,-93.16514618,-90.75571272])
        initial_phase = initial_phase/180*np.pi
        initial_phase = initial_phase.reshape((16, 1))
        music_worker_2D = aoa.Music()
        phase = -phaseS[0, 1:17]
        phase = phase.reshape((16, 1))+initial_phase
        phase2 = -phaseS[0, 34:50]
        phase2 = phase2.reshape((16, 1))+initial_phase
        phase = np.hstack((phase, phase2))
        signal = np.exp(1j * phase)
        psp, azimuth, elevation, intensity = music_worker_2D.music(signal, channel)


    # print(phaseS)
    # plt.plot(phaseS[0])
    # plt.show()
    #music_worker = aoa_1d.AOA_1D()

    # pola = 1 # 0 V or 1 H
    # music_worker_2D = aoa.Music()
    # phase = -phaseS[0, 1+pola:32+pola:2]
    # phase = phase.reshape((16, 1))
    # phase2 = -phaseS[0, 34+pola :65+pola:2]
    # phase2 = phase2.reshape((16, 1))
    # phase = np.hstack((phase,phase2))
    # signal = np.exp(1j * phase)
    # # psp, azimuth, elevation = music_worker.music(signal)
    # # print(azimuth * 180 / np.pi, elevation * 180 / np.pi)
    #
    # #psp, azimuth, intensity = music_worker.music(signal,channel)
    # psp, azimuth, elevation,intensity=music_worker_2D.music(signal,channel)
    # # temp = phaseS[0,0:66]
    # # temp = temp.reshape((2,33))
    print(channel,azimuth * 180 / np.pi,elevation* 180 / np.pi,intensity)
    #print(channel, azimuth * 180 / np.pi, intensity)

    # psp, azimuth, intensity = music_worker.p0(phase)
    # print(azimuth)

    return psp, psp2, azimuth * 180 / np.pi, elevation * 180 / np.pi, intensity
    # phaseS[0, 2] - phaseS[0, 1]