import numpy as np
import time

class Inputs:
    def __init__(self, ux, uy, fx, fy, mx, my):
        self.ux = ux
        self.uy = uy
        self.fx = fx
        self.fy = fy
        self.mx = mx
        self.my = my
        self.Tx = 0.0
        self.Ty = 0.0

class Helper:
    @staticmethod
    def pressure(P, S, x0, x):
        return P - S * (x0 - x)

    @staticmethod
    def force(args_T, AR, P):
        return args_T + AR * P

class Subroutine:
    def __init__(self, args):
        self.args = args

    def sr_v1(self, dy, y):
        Px, Py = 0.0, 0.0
        x0 = np.sqrt(1.0 - y ** 2)
        dx = 2.0 * x0 / self.args.mx
        AR = dx * dy
        B = x0
        x = x0 - dx / 2.0
        Sx = self.args.ux - y * self.args.fx

        while -(x + B) < 0.0:
            Sy = self.args.uy + self.args.fy * (x0 + x) / 2.0
            Px = Helper.pressure(Px, Sx, x0, x)
            Py = Helper.pressure(Py, Sy, x0, x)
            P = np.sqrt(Px ** 2 + Py ** 2) / (1.0 - y ** 2 - x ** 2)

            if P > 1.0:
                Px /= P
                Py /= P
            
            self.args.Tx = Helper.force(self.args.Tx, AR, Px)
            self.args.Ty = Helper.force(self.args.Ty, AR, Py)
            x0 = x
            x -= dx

class Fastsim(Subroutine):
    def __init__(self, args):
        super().__init__(args)
        self.T = []

    def v1(self, dy, TOL):
        s = 1.0
        while s > -2.0:
            if abs(self.args.ux) > abs(self.args.fx) or abs(self.args.uy) > abs(self.args.fy):
                control = False
                dy = 2.0 / self.args.my
                s = -1.0
                ymi = -1.0
            else:
                control = True
                dy = (1.0 - (self.args.ux / self.args.fx * s)) / (
                    np.floor((1.0 - (self.args.ux / self.args.fx * s)) * self.args.my / 2.0) + 1.0)
                ymi = (self.args.ux / self.args.fx * s) + dy

            nk = 1
            y = (1.0 + dy / 2.0) * s
            while nk != 0:
                y -= dy * s
                if y * s < ymi and not control:
                    nk = 0
                elif y * s < ymi:
                    if TOL < (dy / 2.0):
                        dy /= 2.0
                        y += (dy / 2.0) * s
                        self.sr_v1(dy, y)
                        nk = 1
                    else:
                        self.sr_v1(dy, y)
                        nk = 0
                else:
                    self.sr_v1(dy, y)
            s -= 2.0

        self.T = [2.0 * -self.args.Tx / np.pi, 2.0 * self.args.Ty / np.pi]

if __name__ == "__main__":
    ux, uy, fx, fy, mx, my = 1.0, -2.0, 2.0, 1.0, 5.0, 5.0
    args = Inputs(ux, uy, fx, fy, mx, my)
    fastsim = Fastsim(args)

    start_time = time.time()
    fastsim.v1(0.4, 0.09)
    end_time = time.time()

    elapsed_time = (end_time - start_time) * 1e6  # Convert to microseconds
    print(f"Time elapsed: {elapsed_time:.2f} microseconds")
    print(f"Fx= {fastsim.T[0]}")
    print(f"Fy= {fastsim.T[1]}")
