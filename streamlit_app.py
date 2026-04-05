
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from krogh_solver import krogh_solver

st.set_page_config(layout="wide")

col1, col2 = st.columns([1,1])
with col1:
    st.image("image.png")

st.title("Cancer tissue advection-diffusion-reaction model for pH and oxygen dynamics")

st.markdown('[Using Mathematical Modeling of Tumor Metabolism to Predict the Magnitude, Composition, and Hypoxic Interactions of Microenvironment Acidosis](https://onlinelibrary.wiley.com/doi/10.1002/bies.70101)')

st.sidebar.header("Inputs")
L = st.sidebar.number_input("Tissue length (um)", value=2000.0)
R = st.sidebar.number_input("Radius (um)", value=200.0)
RR = st.sidebar.number_input("Respiratory rate (mM/min)", value=1.0)
GR = st.sidebar.number_input("Fermentative rate (mM/min)", value=1.0)
ve = st.sidebar.number_input("Extracellular volume fraction", value=0.2)

startO2 = st.sidebar.number_input("Blood [O2] (mM)", value=0.13)
startCO2 = st.sidebar.number_input("Blood [CO2] (mM)", value=1.2)
startHCO3 = st.sidebar.number_input("Blood [HCO3-] (mM)", value=24.0)
startGlucose = st.sidebar.number_input("Blood [Glucose] (mM)", value=5.0)

CA = st.sidebar.number_input("Tissue CA activity", value=100.0)
pHi0 = st.sidebar.number_input("Initial intracellular pH", value=7.2)

NHE = st.sidebar.radio("NHE activity", ["yes","no"])
n_points = st.sidebar.number_input("Mesh points", value=20)

if st.button("Solve"):

    out = krogh_solver(
        Rtis=R,
        RR=RR,
        GR=GR,
        ve=ve,
        startO2=startO2,
        startCO2=startCO2,
        startHCO3=startHCO3,
        startGlucose=startGlucose,
        CA=CA,
        pHi0=pHi0,
        NHE=NHE,
        Nx=int(n_points),
        L=L
    )

    x = out["x_um"]

    # -------------------------------
    # Data table (clean + consistent)
    # -------------------------------
    df = pd.DataFrame({
        "x (um)": x,
        "Blood O2": out["O2_b"],
        "Blood CO2": out["CO2_b"],
        "Blood HCO3": out["HCO3_b"],
        "Blood pH": out["pHb"],
        "Blood Lac": out["Lac_b"],
        "Blood Glu": out["Glu_b"],

        "Tissue O2": out["O2_t"],
        "Tissue CO2": out["CO2_t"],
        "HCO3e": out["HCO3_e"],
        "HCO3i": out["HCO3_i"],
        "pHe": out["pHe"],
        "pHi": out["pHi"],
        "Lace": out["Lac_e"],
        "Laci": out["Lac_i"],
        "HLac": out["HLac_t"],
        "Glu_t": out["Glu_t"]
    })

    # -------------------------------
    # MATLAB-style multi-panel plots
    # -------------------------------
    fig, axs = plt.subplots(3,4, figsize=(18,12))

    # --- BLOOD ---
    axs[0,0].plot(x, out["O2_b"]); axs[0,0].set_title("Blood O2 (mM)")
    axs[0,1].plot(x, out["CO2_b"]); axs[0,1].set_title("Blood CO2 (mM)")
    axs[0,2].plot(x, out["HCO3_b"]); axs[0,2].set_title("Blood HCO3 (mM)")
    axs[0,3].plot(x, out["pHb"]); axs[0,3].set_title("Blood pH")

    # --- TISSUE OXYGEN + CO2 ---
    axs[1,0].plot(x, out["O2_t"]); axs[1,0].set_title("Tissue O2 (mM)")
    axs[1,1].plot(x, out["CO2_t"]); axs[1,1].set_title("Tissue CO2 (mM)")

    # --- BICARBONATE ---
    axs[1,2].plot(x, out["HCO3_e"], label="Extracellular")
    axs[1,2].plot(x, out["HCO3_i"], '--', label="Intracellular")
    axs[1,2].set_title("HCO3 (mM)")
    axs[1,2].legend()

    # --- pH ---
    axs[1,3].plot(x, out["pHe"], label="Extracellular")
    axs[1,3].plot(x, out["pHi"], '--', label="Intracellular")
    axs[1,3].set_title("pH")
    axs[1,3].legend()

    # --- LACTATE ---
    axs[2,0].plot(x, out["Lac_e"], label="Extracellular")
    axs[2,0].plot(x, out["Lac_i"], '--', label="Intracellular")
    axs[2,0].set_title("Lactate (mM)")
    axs[2,0].legend()

    # --- HLac ---
    axs[2,1].plot(x, out["HLac"]); axs[2,1].set_title("HLac (mM)")

    # --- GLUCOSE ---
    axs[2,2].plot(x, out["Glu_b"], label="Blood")
    axs[2,2].plot(x, out["Glu_t"], '--', label="Tissue")
    axs[2,2].set_title("Glucose (mM)")
    axs[2,2].legend()

    # --- pH vs O2 ---
    axs[2,3].plot(out["O2_t"], out["pHe"], label="Extracellular")
    axs[2,3].plot(out["O2_t"], out["pHi"], '--', label="Intracellular")
    axs[2,3].set_title("pH vs O2")
    axs[2,3].set_xlabel("O2 (mM)")
    axs[2,3].legend()

    # formatting
    for ax in axs.flat:
        ax.set_xlabel("x (um)")
        ax.grid(True)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    st.subheader("Spatial data")
    st.dataframe(df, use_container_width=True)