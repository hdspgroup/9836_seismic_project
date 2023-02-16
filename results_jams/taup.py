import numpy as np
from scipy.sparse import identity
from scipy.io import loadmat, savemat


class Tau_pTransform:
    """
    Perform the tau-p transform over a gather shot.
    input:
         x: a numpy array with the positions of the receiver (from 0 to max_x) (row array)
         t: a numpy array with the time recorded (from 0 to max_t) (row array)
         dx: sample rate in the offset
         dt: sample rate in the temporal axis
         slowness: a numpy array with the reciprocal of the velocities (1/v)
    """

    def __init__(self, x, t, dx, dt, slowness, mu=0.01):
        self.slowness = slowness
        self.lp = len(slowness.T)
        self.x = x
        self.dx = dx
        self.lx = len(x.T)
        self.t = t
        self.dt = dt
        self.lt = len(t.T)
        pm = np.max(self.slowness)
        self.IF = [0] * self.lt
        self.Gi = [0] * self.lt
        self.I = identity(self.lx) * mu
        self.grad = np.zeros((self.lp, self.lt)).astype(complex)
        print(self.lt)
        print("generando matrices\n -------------------------------------------")
        Di = np.zeros((1, self.lp)).astype(complex)
        for i in range(self.lt):
            print("matrix {0} de {1}".format(i, self.lt - 1))
            G = np.exp(-2j * np.pi * ((i / self.lt) * (1 / self.dt)) * x.T.dot(self.slowness))
            Di[:] = np.exp(
                -1j * np.pi * (i / self.lt) * (1.0 / self.dt) * 1.0 * np.sqrt(pm ** 2 - self.slowness ** 2)) - np.exp(
                1j * np.pi * (i / self.lt) * (1.0 / self.dt) * np.sqrt(pm ** 2 - self.slowness ** 2))
            for k in range(self.lx):
                G[k, :] = G[k, :] * Di
            self.IF[i] = np.linalg.pinv(G.dot(np.conj(G.T)) + self.I)
            self.Gi[i] = G
        print("finished")

    def inverse(self):
        def transform(X):
            for i in range(X.shape[1]):
                self.grad[:, i] = (np.conj(self.Gi[i].T).dot(self.IF[i])).dot(X[:, i])
            return self.grad.copy()

        return transform

    def direct(self):
        def transform(X):
            resid = np.zeros((self.lx, self.lt)).astype(complex)
            for i in range(X.shape[1]):
                resid[:, i] = self.Gi[i].dot(X[:, i])
            return resid

        return transform


data = loadmat('datocomprimido.mat')
data = data['data']
print(data.shape)
dm = int(0.003 * 3000)
a = 1 + (((np.linspace(0, 9000, 1001)) / 1000) ** 2)
a = (a / np.max(a)) * 4
a = a + 1
print(dm)
data1 = data.copy()
nt = 1001
for k in range(nt):
    data1[k, :] = data[k, :] * a[k]
data1 = data1 / np.max(np.abs(data1))
shot1o = data1
shot1r = shot1o[:, 24:]
dx = 25
dt = 0.003
nx = 56
x = np.linspace(0, nx-1, nx) * dx
x = np.expand_dims(x, -1)
x = x.T
t = np.linspace(0, 3000, 1001)
t = np.expand_dims(t, -1)
t = t.T
s = np.linspace(800, 4900, 1026)
s = np.expand_dims(s, -1)
s = s.T
s = 1 / s
A = Tau_pTransform(x, t, dx, dt, s)

shot1r = shot1r.T
shot1r = shot1r.astype(complex)
for k in range(shot1r.shape[0]):
    shot1r[k, :] = np.fft.fft(shot1r[k, :])
savemat('shot1r.mat', {'shot1r': shot1r})
trp = A.inverse()
drt = A.direct()
shoten = trp(shot1r)
savemat('shoten.mat', {'shoten': shoten})
shotre = drt(shoten)
savemat('shotre.mat', {'shotre': shotre})
