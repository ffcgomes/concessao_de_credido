import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Função para carregar modelo e encoder
@st.cache_resource
def load_model_encoder():
    model = joblib.load("./resultados_parciais/modelo_logistico.pkl")
    encoder = joblib.load("./resultados_parciais/encoder.pkl")
    return model, encoder

model, encoder = load_model_encoder()

# Limites dos sliders
limits = {
    "ValorQuitacao": (10., 10000.),
    "Atraso": (2.0, 24.0),
    "Quant_Pagamentos_Via_Boleto": (0, 15),
    "Quant_Ocorrencia": (18, 90)
}

# Título
st.title("Concessão de Crédito")

# Estilo
st.markdown("""
    <style>
        .stSlider label, .stSelectbox label {
            font-size: 18px !important;
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Entradas do usuário
valor_quitacao = st.slider("**Valor de Parcela**", *limits["ValorQuitacao"], value=(limits["ValorQuitacao"][0] + limits["ValorQuitacao"][1]) / 2, step=0.01)
atraso = st.slider("**Nº de Parcelas**", *limits["Atraso"], value=(limits["Atraso"][0] + limits["Atraso"][1]) / 2, step=1.0)
quant_pagamentos_via_boleto = st.slider("**Quant. de Boletos pagos**", *limits["Quant_Pagamentos_Via_Boleto"], value=(limits["Quant_Pagamentos_Via_Boleto"][0] + limits["Quant_Pagamentos_Via_Boleto"][1]) // 2, step=1)
quant_ocorrencia = st.slider("**Idade do Cliente**", *limits["Quant_Ocorrencia"], value=(limits["Quant_Ocorrencia"][0] + limits["Quant_Ocorrencia"][1]) // 2, step=1)
uf = st.selectbox("**UF**", list(encoder.categories_[0]))

# DataFrame usuário
user_data = pd.DataFrame({
    "ValorQuitacao": [valor_quitacao],
    "Atraso": [atraso],
    "Quant_Pagamentos_Via_Boleto": [quant_pagamentos_via_boleto],
    "Quant_Ocorrencia": [quant_ocorrencia],
    "UF": [uf]
})

# Codificação da UF
uf_encoded_user = encoder.transform(user_data[["UF"]])
uf_encoded_user_df = pd.DataFrame(
    uf_encoded_user.toarray(), 
    columns=encoder.get_feature_names_out(["UF"])
)

# Junta tudo
user_data = user_data.drop(columns=["UF"]).reset_index(drop=True)
user_data = pd.concat([user_data, uf_encoded_user_df], axis=1)
user_data = user_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Previsão
probabilidade = model.predict_proba(user_data)[:, 1][0]

# Resultado
st.subheader(f"Probabilidade de pagamento: **{probabilidade:.3f}**")
if probabilidade > 0.3:
    st.header("Cliente apto a receber crédito")
else:
    st.header("Cliente não apto a receber crédito")

if probabilidade > 0.5:
    st.success("O cliente tem alta probabilidade de pagar a dívida.")
else:
    st.error("O cliente tem baixa probabilidade de pagar a dívida.")

# Rodapé
st.write("Desenvolvido com ❤️ usando Streamlit")
