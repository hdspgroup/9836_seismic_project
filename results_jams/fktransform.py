import numpy


def fktransform(data: numpy,dt: float,dx: float):
    nt=data.shape[0]
    nx=data.shape[1]
    nt_fft=2*nt
    nx_fft=2*nx
    data_f = numpy.fft.fft(data,nt_fft,0)
    data_fk = numpy.fft.fft(data_f,nx_fft,1)
    FK=numpy.fft.fftshift(numpy.abs((data_fk)))#20*log10
    FK=FK[nt:, :]
    f = numpy.linspace(-0.5,0.5,nt_fft)/dt
    kx = numpy.linspace(-0.5,0.5,nt_fft)/dx
    return FK, f, kx