import numpy as np
from scipy.integrate import solve_ivp

def krogh_solver(
    Rtis, RR, GR, ve,
    startO2, startCO2, startHCO3, startGlucose,
    CA, pHi0, NHE, Nx, L
):

    # --- geometry ---
    Rcap = 5.0
    L = float(L)
    v = 1000.0

    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]

    vi = 1 - ve

    # --- transport ---
    D_free = np.array([2600,2100,1300,10,1000,1000,960,0,0,0], dtype=float)
    D_t = np.concatenate([D_free[:2], D_free[2:] * ve])
    h = D_t / (Rcap * np.log((Rtis + Rcap)/Rcap))

    SV_b = 2.0 / Rcap
    SV_t = 2.0 * Rcap / ((Rcap + Rtis)**2 - Rcap**2)

    K_b = h * SV_b
    K_t = h * SV_t

    # --- reactions ---
    CA_bld = CA
    CA_tis = CA

    kh = 0.14
    kr = kh / (10**-6.1)

    kf = 1e6
    kb = kf / (10**-3.9)

    JR = (RR/1000)/60
    JG = (GR/1000)/60

    KmO2 = 1e-6
    KgG = 1e-3

    # --- inlet ---
    pH_b = 7.4
    H_b = 10**(-pH_b)

    B_in = np.array([
        startO2/1000,
        startCO2/1000,
        startHCO3/1000,
        H_b,
        0,
        0,
        startGlucose/1000
    ])

    # --- initial ---
    U0 = np.zeros((Nx,17))
    U0[:,0:7] = B_in
    U0[:,7:14] = B_in
    U0[:,14] = startHCO3/1000
    U0[:,15] = 10**(-pHi0)
    U0[:,16] = 0

    y0 = U0.flatten()

    # ---------- RHS ----------
    def rhs(t, y):

        U = y.reshape(Nx,17)

        # --- upwind gradient ---
        dUdx = np.zeros_like(U)
        dUdx[1:] = (U[1:] - U[:-1]) / dx

        B = U[:,0:7]
        T = U[:,7:17]

        O2_b, CO2_b, HCO3_b, H_b, Lac_b, HLac_b, Glu_b = B.T
        O2_t, CO2_t, HCO3_e, H_e, Lac_e, HLac_t, Glu_t, HCO3_i, H_i, Lac_i = T.T

        # --- reactions (vectorised) ---
        rCO2_b = CA_bld*(kr*HCO3_b*H_b - kh*CO2_b)
        rHLac_b = kb*Lac_b*H_b - kf*HLac_b

        rCO2_e = CA_tis*(kr*HCO3_e*H_e - kh*CO2_t)
        rHLac_e = kb*Lac_e*H_e - kf*HLac_t

        rCO2_i = CA_tis*(kr*HCO3_i*H_i - kh*CO2_t)
        rHLac_i = kb*Lac_i*H_i - kf*HLac_t

        # --- metabolism ---
        O2eff = np.maximum(O2_t,0)
        mmO2 = O2eff/(O2eff + KmO2)

        Glueff = np.maximum(Glu_t,0)
        mmG = Glueff/(Glueff + KgG)

        mmH = (10**-7.1)**2.25 / (np.maximum(H_i,1e-12)**2.25 + (10**-7.1)**2.25)

        Jresp = JR*mmG*mmO2
        Jglyc = JG*mmG*mmH

        # --- BLOOD ---
        s_b = np.zeros_like(B)

        s_b[:,0] = K_b[0]*(O2_t - O2_b) - v*dUdx[:,0]
        s_b[:,1] = K_b[1]*(CO2_t - CO2_b) - v*dUdx[:,1] + rCO2_b
        s_b[:,2] = K_b[2]*(HCO3_e - HCO3_b) - v*dUdx[:,2] - rCO2_b
        s_b[:,3] = K_b[3]*(H_e - H_b) - v*dUdx[:,3] - rCO2_b - rHLac_b
        s_b[:,4] = K_b[4]*(Lac_e - Lac_b) - v*dUdx[:,4] - rHLac_b
        s_b[:,5] = K_b[5]*(HLac_t - HLac_b) - v*dUdx[:,5] + rHLac_b
        s_b[:,6] = K_b[6]*(Glu_t - Glu_b) - v*dUdx[:,6]

        # --- TISSUE ---
        s_t = np.zeros_like(T)

        s_t[:,0] = K_t[0]*(O2_b - O2_t) - 6*vi*Jresp
        s_t[:,1] = K_t[1]*(CO2_b - CO2_t) + 6*vi*Jresp + vi*rCO2_i + ve*rCO2_e
        s_t[:,2] = K_t[2]*(HCO3_b - HCO3_e) - ve*rCO2_e
        s_t[:,3] = K_t[3]*(H_b - H_e) - ve*rCO2_e - ve*rHLac_e
        s_t[:,4] = K_t[4]*(Lac_b - Lac_e) - ve*rHLac_e
        s_t[:,5] = K_t[5]*(HLac_b - HLac_t) + 2*vi*Jglyc + ve*rHLac_e + vi*rHLac_i
        s_t[:,6] = K_t[6]*(Glu_b - Glu_t) - vi*(Jglyc + Jresp)
        s_t[:,7] = -vi*rCO2_i
        s_t[:,8] = -vi*rCO2_i - vi*rHLac_i
        s_t[:,9] = -vi*rHLac_i

        dUdt = np.hstack([s_b, s_t])

        return dUdt.flatten()

    # ---------- steady-state stopping ----------
    def steady_event(t, y):
        dydt = rhs(t,y)
        return np.max(np.abs(dydt)) - 1e-6

    steady_event.terminal = True
    steady_event.direction = -1

    # ---------- tolerances ----------
    atol = np.ones(17*Nx)*1e-6

    # tighter for H+
    for i in range(Nx):
        atol[i*17 + 3] = 1e-12      # blood H+
        atol[i*17 + 10] = 1e-12     # extracellular H+
        atol[i*17 + 15] = 1e-12     # intracellular H+

    sol = solve_ivp(
        rhs,
        [0, 20000],
        y0,
        method="BDF",
        atol=atol,
        rtol=1e-6,
        events=steady_event
    )

    U = sol.y[:,-1].reshape(Nx,17)

    return {
        "x_um": x,
        "O2_b": 1e3*U[:,0],
        "CO2_b": 1e3*U[:,1],
        "HCO3_b": 1e3*U[:,2],
        "pHb": -np.log10(U[:,3]),
        "Lac_b": 1e3*U[:,4],
        "Glu_b": 1e3*U[:,6],

        "O2_t": 1e3*U[:,7],
        "CO2_t": 1e3*U[:,8],
        "HCO3_e": 1e3*U[:,9],
        "pHe": -np.log10(U[:,10]),
        "Lac_e": 1e3*U[:,11],
        "HLac_t": 1e3*U[:,12],
        "Glu_t": 1e3*U[:,13],
        "HCO3_i": 1e3*U[:,14],
        "pHi": -np.log10(U[:,15]),
        "Lac_i": 1e3*U[:,16],
    }