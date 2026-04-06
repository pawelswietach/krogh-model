
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from krogh_solver import krogh_solver

st.set_page_config(layout="wide")

st.title("Cancer Tissue Transport Model")

st.sidebar.header("Inputs")

L = st.sidebar.number_input("Length (µm)", value=2000.0)
R = st.sidebar.number_input("Radius (µm)", value=200.0)

RR = st.sidebar.number_input("Respiration (mM/min)", value=1.0)
GR = st.sidebar.number_input("Glycolysis (mM/min)", value=1.0)

ve = st.sidebar.number_input("Extracellular fraction", value=0.2)

startO2 = st.sidebar.number_input("O2 (mM)", value=0.13)
startCO2 = st.sidebar.number_input("CO2 (mM)", value=1.2)
startHCO3 = st.sidebar.number_input("HCO3 (mM)", value=24.0)
startGlucose = st.sidebar.number_input("Glucose (mM)", value=5.0)

CA = st.sidebar.number_input("CA activity", value=100.0)
pHi0 = st.sidebar.number_input("Initial pHi", value=7.2)

NHE = 1.0 if st.sidebar.radio("NHE", ["yes","no"]) == "yes" else 0.0
Nx = int(st.sidebar.number_input("Mesh points", value=50))

if st.button("Solve"):

    with st.spinner("Solving to steady state..."):
        out = krogh_solver(R, RR, GR, ve,
                           startO2, startCO2, startHCO3, startGlucose,
                           CA, pHi0, NHE, Nx, L)

    x = out["x_um"]

    fig, ax = plt.subplots()
    ax.plot(x, out["O2_b"], label="Blood O2")
    ax.plot(x, out["O2_t"], label="Tissue O2")
    ax.legend()
    ax.set_xlabel("x (µm)")
    ax.grid(True)

    st.pyplot(fig)

    df = pd.DataFrame(out)
    st.dataframe(df)
