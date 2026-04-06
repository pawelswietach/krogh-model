import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from krogh_solver import krogh_solver

st.set_page_config(layout="wide")

# Header
col1, col2 = st.columns([1,1])
with col1:
    st.image("image.png")

st.title("Cancer tissue advection-diffusion-reaction model for pH and oxygen dynamics")

st.markdown('[Using Mathematical Modeling of Tumor Metabolism to Predict the Magnitude, Composition, and Hypoxic Interactions of Microenvironment Acidosis](https://onlinelibrary.wiley.com/doi/10.1002/bies.70101)')


# Sidebar
st.sidebar.header("Inputs")

L = st.sidebar.number_input("Tissue length (µm)", value=2000.0)
R = st.sidebar.number_input("Tissue radius (µm)", value=200.0)
RR = st.sidebar.number_input("Respiratory rate (mM/min)", value=0.1)
GR = st.sidebar.number_input("Fermentative rate (mM/min)", value=0.5)
ve = st.sidebar.number_input("Extracellular fraction", value=0.25)

startO2 = st.sidebar.number_input("Blood O2 (mM)", value=0.13)
startCO2 = st.sidebar.number_input("Blood CO2 (mM)", value=1.2)
startHCO3 = st.sidebar.number_input("Blood HCO3- (mM)", value=24.0)
startGlucose = st.sidebar.number_input("Blood Glucose (mM)", value=5.0)

CA = st.sidebar.number_input("Tissue CA activity", value=100.0)
pHi0 = st.sidebar.number_input("Initial pHi", value=7.2)

NHE = 1.0 if st.sidebar.radio("NHE", ["yes","no"])=="yes" else 0.0
Nx = int(st.sidebar.number_input("Mesh points", value=20))

if st.button("Solve"):

    out = krogh_solver(R,RR,GR,ve,
                       startO2,startCO2,startHCO3,startGlucose,
                       CA,pHi0,NHE,Nx,L)

    x = out["x"]

    fig, axs = plt.subplots(3,4, figsize=(18,12))

    # TOP ROW
    axs[0,0].plot(x,out["O2_b"],'r', label="Blood")
    axs[0,0].plot(x,out["O2_t"],'k', label="Tissue")
    axs[0,0].set_title("O2 (mM)")
    axs[0,0].legend()

    axs[0,1].plot(x,out["Glu_b"],'r', label="Blood")
    axs[0,1].plot(x,out["Glu_t"],'k', label="Tissue")
    axs[0,1].set_title("Glucose (mM)")
    axs[0,1].legend()

    axs[0,2].plot(x,out["CO2_b"],'r', label="Blood")
    axs[0,2].plot(x,out["CO2_t"],'k', label="Tissue")
    axs[0,2].set_title("CO2 (mM)")
    axs[0,2].legend()

    axs[0,3].plot(x,1000*out["HLac_b"],'r', label="Blood")
    axs[0,3].plot(x,1000*out["HLac_t"],'k', label="Tissue")
    axs[0,3].set_title("HLac (µM)")
    axs[0,3].legend()

    # BOTTOM ROW
    axs[1,0].plot(x,out["HCO3_b"],'r', label="Blood")
    axs[1,0].plot(x,out["HCO3_i"],color='orange', label="Extracellular")
    axs[1,0].plot(x,out["HCO3_e"],'b', label="Intracellular")
    axs[1,0].set_title("Bicarbonate (mM)")
    axs[1,0].legend()

    axs[1,1].plot(x,out["pHb"],'r', label="Blood")
    axs[1,1].plot(x,out["pHi"],color='orange', label="Extracellular")
    axs[1,1].plot(x,out["pHe"],'b', label="Intracellular")
    axs[1,1].set_title("pH")
    axs[1,1].legend()

    axs[1,2].plot(x,out["Lac_b"],'r', label="Blood")
    axs[1,2].plot(x,out["Lac_i"],color='orange', label="Extracellular")
    axs[1,2].plot(x,out["Lac_e"],'b', label="Intracellular")
    axs[1,2].set_title("Lactate (mM)")
    axs[1,2].legend()

    # pH vs O2
    axs[1,3].plot(out["O2_t"],out["pHe"],color='orange', label="Extracellular")
    axs[1,3].plot(out["O2_t"],out["pHi"],'b', label="Intracellular")
    axs[1,3].set_title("pH vs O2")
    axs[1,3].legend()

    for i, ax in enumerate(axs.flat):
        if i != 7:
            ax.set_xlabel("Length along capillary (µm)")


    plt.subplots_adjust(hspace=0.5)

    st.pyplot(fig, use_container_width=True)

    st.subheader("Spatial data")
    st.dataframe(df, use_container_width=True)