
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from krogh_solver import krogh_solver

st.set_page_config(layout="wide")

col1, col2 = st.columns([1,1])
with col1:
    st.image("image.png")

st.title("Cancer spheroid diffusion-reaction model for pH and oxygen dynamics")

st.markdown('[Using Mathematical Modeling of Tumor Metabolism to Predict the Magnitude, Composition, and Hypoxic Interactions of Microenvironment Acidosis](https://onlinelibrary.wiley.com/doi/10.1002/bies.70101)')

st.sidebar.header("Inputs")

R = st.sidebar.number_input("Radius (um)", value=200.0)
RR = st.sidebar.number_input("Respiratory rate (mM/min)", value=1.0)
GR = st.sidebar.number_input("Fermentative rate (mM/min)", value=1.0)
ve = st.sidebar.number_input("Extracellular volume fraction", value=0.2)

startO2 = st.sidebar.number_input("Bath [O2] (mM)", value=0.13)
startCO2 = st.sidebar.number_input("Bath [CO2] (mM)", value=1.2)
startHCO3 = st.sidebar.number_input("Bath [HCO3-] (mM)", value=24.0)
startGlucose = st.sidebar.number_input("Bath [Glucose] (mM)", value=5.0)

CA = st.sidebar.number_input("Carbonic anhydrase activity", value=100.0)
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
        Nx=int(n_points)
    )
    x = out["x_um"]
    depth = R - x

    df = pd.DataFrame({
        "x (um)": x,
        "O2 (mM)": out["O2_mM"],
        "CO2 (mM)": out["CO2_mM"],
        "HCO3 (mM)": out["HCO3_mM"],
        "pH": out["pH"],
    })

    fig, ax = plt.subplots(1,3, figsize=(12,4))

    ax[0].plot(out["x_um"], out["O2_mM"])
    ax[0].set_title("O2")
 
    ax[1].plot(out["x_um"], out["CO2_mM"])
    ax[1].set_title("CO2")

    ax[2].plot(out["x_um"], out["pH"])
    ax[2].set_title("pH")

    plt.subplots_adjust(hspace=0.5)

    st.pyplot(fig, use_container_width=True)

    st.subheader("Spatial data")
    st.dataframe(df, use_container_width=True)
