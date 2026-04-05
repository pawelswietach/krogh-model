import numpy as np
from scipy.integrate import solve_ivp

def krogh_solver(Rtis=200, RR=1, GR=1, NHE="yes", Nx=50):

    # --- geometry ---
    Rcap = 5
    L = 2000
    v = 1000

    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]

    # --- parameters ---
    vi = 0.75
    ve = 1 - vi

    CA_bld = 1000
    CA_tis = 100

    kh = 0.14
    kr = kh / (10**-6.1)

    kf = 1e6
    kb = kf / (10**-3.9)

    JR = (RR/1000)/60
    JG = (GR/1000)/60

    KmO2 = 1e-6
    KgG = 1e-3

    TBuf = 38.5/1000
    THb = 2.33*4/1000

    # --- inlet ---
    pH_b = 7.4
    O2_b = 0.13/1000
    CO2_b = 1.2/1000
    H_b = 10**(-pH_b)
    HCO3_b = 10**(pH_b-6.1)*CO2_b

    Lac_b = 0
    HLac_b = 0
    Glu_b = 5/1000

    B_in = np.array([O2_b, CO2_b, HCO3_b, H_b, Lac_b, HLac_b, Glu_b])

    # --- initial state ---
    U0 = np.zeros((Nx,17))
    for i in range(Nx):
        U0[i,0:7] = B_in
        U0[i,7:14] = B_in
        U0[i,14] = HCO3_b
        U0[i,15] = H_b
        U0[i,16] = 0

    y0 = U0.flatten()

    # --- RHS ---
    def rhs(t, y):
        U = y.reshape(Nx,17)
        dUdt = np.zeros_like(U)

        for i in range(Nx):

            if i == 0:
                dUdx = np.zeros(17)
            else:
                dUdx = (U[i] - U[i-1]) / dx

            B = U[i,0:7]
            T = U[i,7:17]

            O2_b, CO2_b, HCO3_b, H_b, Lac_b, HLac_b, Glu_b = B
            O2_t, CO2_t, HCO3_e, H_e, Lac_e, HLac_t, Glu_t, HCO3_i, H_i, Lac_i = T

            # reactions
            rCO2_b = CA_bld*(kr*HCO3_b*H_b - kh*CO2_b)
            rHLac_b = kb*Lac_b*H_b - kf*HLac_b

            # simple advection + reaction only (minimal version)
            s_b = np.zeros(7)

            s_b[1] = -v*dUdx[1] + rCO2_b
            s_b[2] = -v*dUdx[2] - rCO2_b
            s_b[3] = -v*dUdx[3] - rCO2_b - rHLac_b

            dUdt[i,0:7] = s_b

        return dUdt.flatten()

    sol = solve_ivp(rhs, [0,2000], y0, method='BDF')

    U = sol.y[:,-1].reshape(Nx,17)

    return {
        "x_um": x,
        "O2_mM": 1e3*U[:,0],
        "CO2_mM": 1e3*U[:,1],
        "HCO3_mM": 1e3*U[:,2],
        "pH": -np.log10(U[:,3])
    }