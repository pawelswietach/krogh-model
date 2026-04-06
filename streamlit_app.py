import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from krogh_solver import krogh_solver

st.set_page_config(layout="wide")

# -------------------------------
# Header / image
# -------------------------------
col1, col2 = st.columns([1,1])
with col1:
    st.image("image.png")

st.title("Cancer tissue advection-diffusion-reaction model")

st.markdown(
    "[Mathematical Modeling of Tumor Metabolism](https://onlinelibrary.wiley.com/doi/10.1002/bies.70101)"
)

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("Inputs")

L = st.sidebar.number_input("Tissue length (um)", value=2000.0)
R = st.sidebar.number_input("Radius (um)", value=200.0)

RR = st.sidebar.number_input("Respiration rate (mM/min)", value=1.0)
GR = st.sidebar.number_input("Glycolysis rate (mM/min)", value=1.0)

ve = st.sidebar.number_input("Extracellular volume fraction", value=0.2)

startO2 = st.sidebar.number_input("Blood [O2] (mM)", value=0.13)
startCO2 = st.sidebar.number_input("Blood [CO2] (mM)", value=1.2)
startHCO3 = st.sidebar.number_input("Blood [HCO3-] (mM)", value=24.0)
startGlucose = st.sidebar.number_input("Blood [Glucose] (mM)", value=5.0)

CA = st.sidebar.number_input("Carbonic anhydrase activity", value=100.0)
pHi0 = st.sidebar.number_input("Initial intracellular pH", value=7.2)

NHE = st.sidebar.radio("NHE activity", ["yes","no"])

n_points = st.sidebar.number_input("Mesh points", value=50)

# -------------------------------
# Solve button
# -------------------------------
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
    # Data table
    # -------------------------------
    df = pd.DataFrame({
        "x (um)": x,
        "O2_b": out["O2_b"],
        "CO2_b": out["CO2_b"],
        "HCO3_b": out["HCO3_b"],
        "pHb": out["pHb"],
        "Lac_b": out["Lac_b"],
        "Glu_b": out["Glu_b"],
        "O2_t": out["O2_t"],
        "CO2_t": out["CO2_t"],
        "HCO3_e": out["HCO3_e"],
        "HCO3_i": out["HCO3_i"],
        "pHe": out["pHe"],
        "pHi": out["pHi"],
        "Lac_e": out["Lac_e"],
        "Lac_i": out["Lac_i"],
        "HLac": out["HLac_t"],
        "Glu_t": out["Glu_t"]
    })

    # -------------------------------
    # Plotting (MATLAB-style)
    # -------------------------------
    fig, axs = plt.subplots(3,4, figsize=(18,12))

    # --- O2 ---
    axs[0,0].plot(x, out["O2_b"], 'r', label="Blood")
    axs[0,0].plot(x, out["O2_t"], 'k', label="Tissue")
    axs[0,0].set_title("O2 (mM)")
    axs[0,0].legend()

    # --- CO2 (single, black) ---
    axs[0,1].plot(x, out["CO2_b"], 'k')
    axs[0,1].set_title("CO2 (mM)")

    # --- HCO3 ---
    axs[0,2].plot(x, out["HCO3_b"], 'r', label="Blood")
    axs[0,2].plot(x, out["HCO3_e"], color='orange', label="Extracellular")
    axs[0,2].plot(x, out["HCO3_i"], 'b', label="Intracellular")
    axs[0,2].set_title("HCO3 (mM)")
    axs[0,2].legend()

    # --- pH ---
    axs[0,3].plot(x, out["pHb"], 'r', label="Blood")
    axs[0,3].plot(x, out["pHe"], color='orange', label="Extracellular")
    axs[0,3].plot(x, out["pHi"], 'b', label="Intracellular")
    axs[0,3].set_title("pH")
    axs[0,3].legend()

    # --- Lactate ---
    axs[1,0].plot(x, out["Lac_b"], 'r', label="Blood")
    axs[1,0].plot(x, out["Lac_e"], color='orange', label="Extracellular")
    axs[1,0].plot(x, out["Lac_i"], 'b', label="Intracellular")
    axs[1,0].set_title("Lactate (mM)")
    axs[1,0].legend()

    # --- HLac (single, black) ---
    axs[1,1].plot(x, out["HLac"], 'k')
    axs[1,1].set_title("HLac (mM)")

    # --- Glucose ---
    axs[1,2].plot(x, out["Glu_b"], 'r', label="Blood")
    axs[1,2].plot(x, out["Glu_t"], 'k', label="Tissue")
    axs[1,2].set_title("Glucose (mM)")
    axs[1,2].legend()

    # --- pH vs O2 ---
    axs[1,3].plot(out["O2_t"], out["pHe"], color='orange', label="Extracellular")
    axs[1,3].plot(out["O2_t"], out["pHi"], 'b', label="Intracellular")
    axs[1,3].set_title("pH vs O2")
    axs[1,3].set_xlabel("O2 (mM)")
    axs[1,3].legend()

    # --- empty bottom row ---
    for j in range(4):
        axs[2,j].axis('off')

    # formatting
    for ax in axs.flat:
        if ax.has_data():
            ax.set_xlabel("x (um)")
            ax.grid(True)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    # -------------------------------
    # Data output
    # -------------------------------
    st.subheader("Spatial data")
    st.dataframe(df, use_container_width=True)