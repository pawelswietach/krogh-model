import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from krogh_solver import krogh_solver

st.set_page_config(layout="wide")

# Header
col1, col2 = st.columns([1,1])
with col1:
    st.image("image.png")

st.title("Cancer tissue advection-diffusion-reaction model")

st.markdown(
    "[Mathematical Modeling of Tumor Metabolism](https://onlinelibrary.wiley.com/doi/10.1002/bies.70101)"
)

# Sidebar
st.sidebar.header("Inputs")

L = st.sidebar.number_input("Tissue length (um)", value=2000.0)
R = st.sidebar.number_input("Radius (um)", value=200.0)
RR = st.sidebar.number_input("Respiration rate", value=1.0)
GR = st.sidebar.number_input("Glycolysis rate", value=1.0)
ve = st.sidebar.number_input("Extracellular fraction", value=0.2)

startO2 = st.sidebar.number_input("O2", value=0.13)
startCO2 = st.sidebar.number_input("CO2", value=1.2)
startHCO3 = st.sidebar.number_input("HCO3", value=24.0)
startGlucose = st.sidebar.number_input("Glucose", value=5.0)

CA = st.sidebar.number_input("CA", value=100.0)
pHi0 = st.sidebar.number_input("Initial pHi", value=7.2)

NHE = 1.0 if st.sidebar.radio("NHE", ["yes","no"])=="yes" else 0.0
Nx = int(st.sidebar.number_input("Mesh points", value=10))

if st.button("Solve"):

    out = krogh_solver(R,RR,GR,ve,
                       startO2,startCO2,startHCO3,startGlucose,
                       CA,pHi0,NHE,Nx,L)

    x = out["x"]

    fig, axs = plt.subplots(3,4, figsize=(18,12))

    # TOP ROW
    axs[0,0].plot(x,out["O2_b"],'r')
    axs[0,0].plot(x,out["O2_t"],'k')
    axs[0,0].set_title("O2")

    axs[0,1].plot(x,out["Glu_b"],'r')
    axs[0,1].plot(x,out["Glu_t"],'k')
    axs[0,1].set_title("Glucose")

    axs[0,2].plot(x,out["CO2_b"],'r')
    axs[0,2].plot(x,out["CO2_t"],'k')
    axs[0,2].set_title("CO2")

    axs[0,3].plot(x,out["HLac_b"],'r')
    axs[0,3].plot(x,out["HLac_t"],'k')
    axs[0,3].set_title("HLac")

    # BOTTOM ROW
    axs[1,0].plot(x,out["HCO3_b"],'r')
    axs[1,0].plot(x,out["HCO3_i"],color='orange')
    axs[1,0].plot(x,out["HCO3_e"],'b')
    axs[1,0].set_title("HCO3")

    axs[1,1].plot(x,out["pHb"],'r')
    axs[1,1].plot(x,out["pHi"],color='orange')
    axs[1,1].plot(x,out["pHe"],'b')
    axs[1,1].set_title("pH")

    axs[1,2].plot(x,out["Lac_b"],'r')
    axs[1,2].plot(x,out["Lac_i"],color='orange')
    axs[1,2].plot(x,out["Lac_e"],'b')
    axs[1,2].set_title("Lactate")

    # pH vs O2
    axs[1,3].plot(out["O2_t"],out["pHe"],color='orange')
    axs[1,3].plot(out["O2_t"],out["pHi"],'b')
    axs[1,3].set_title("pH vs O2")

    # empty row
    for j in range(4):
        axs[2,j].axis('off')

    for ax in axs.flat:
        if ax.has_data():
            ax.set_xlabel("x")
            ax.grid(True)

    st.pyplot(fig)

    st.subheader("Data")
    st.dataframe(pd.DataFrame(out))