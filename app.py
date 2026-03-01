import streamlit as st
import joblib
import pandas as pd

# ===============================
# CARREGAR MODELO
# ===============================
# O model.joblib deve ser um PIPELINE já treinado
# (preprocessamento + modelo)
model = joblib.load("model.joblib")

# Caso tenha usado label encoder no treino
# (se não existir, pode remover)
try:
    label_encoder = joblib.load("label_encoder.joblib")
except:
    label_encoder = None


# ===============================
# INTERFACE
# ===============================
st.set_page_config(page_title="Diagnóstico de Obesidade", layout="wide")

st.title("🏥 Sistema de Diagnóstico de Obesidade")
st.markdown("Insira os dados do paciente para realizar a predição.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gênero", ["Female", "Male"])
    age = int(st.number_input("Idade", min_value=14, max_value=61, value=25, step=1))
    height = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
    family_history = st.selectbox("Histórico Familiar de Sobrepeso?", ["yes", "no"])
    favc = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
    fcvc = st.slider("Frequência de consumo de vegetais (1=raramente, 2=às vezes, 3=sempre)", 1, 2, 3)
    ncp = st.slider("Número de refeições principais por dia", 1, 2, 3, 4)

with col2:
    caec = st.selectbox("Consome lanches entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("É fumante?", ["yes", "no"])
    ch2o = st.slider("Consumo diário de água ( 1=< 1 L/dia, 2=1–2 L/dia, 3=> 2 L/dia)", 1, 2, 3)
    scc = st.selectbox("Monitora calorias ingeridas?", ["yes", "no"])
    faf = st.slider("Frequência de atividade física semanal (0=Nenhuma,  1=~1–2×/sem, 2=~3–4×/sem, 3=5×/sem)", 0, 1, 2, 3)
    tue = st.slider("Tempo diário em dispositivos eletrônicos (0=0-2h,  1=~3–5 h/dia. 2=>5h)", 0, 1, 2)
    calc = st.selectbox("Consumo de álcool?", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox(
        "Meio de transporte habitual",
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
    )


# ===============================
# PREDIÇÃO
# ===============================
if st.button("Realizar Diagnóstico"):

    # DataFrame EXACTAMENTE com colunas usadas no treino
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }])

    # Pipeline faz todo o preprocessing sozinho
    prediction = model.predict(input_data)[0]

    # Se existir label encoder
    if label_encoder is not None:
        prediction = label_encoder.inverse_transform([prediction])[0]

    st.success(f"🏥 O nível de obesidade diagnosticado é: **{prediction.replace('_',' ')}**")