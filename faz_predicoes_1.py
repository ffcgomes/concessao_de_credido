import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Carregar o modelo e o encoder
@st.cache_data
def load_model():
    with open("./resultados_parciais/modelo_logistico.pkl", "rb") as f:
        return joblib.load(f)

model, encoder = load_model()

# Definir os valores mínimo e máximo dos atributos com base no arquivo de dados
limits = {
    "ValorQuitacao": (10., 10000.),
    "Atraso": (2.0, 24.0),
    "Quant_Pagamentos_Via_Boleto": (0, 15),
    "Quant_Ocorrencia": (18, 90)
}

# Criar interface do Streamlit
st.title("Concessão de Crédito")

# Aplicar estilo para aumentar a fonte dos labels
st.markdown("""
    <style>
        .stSlider label, .stSelectbox label {
            font-size: 18px !important;
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Entrada do usuário com número e barra deslizante
valor_quitacao = st.slider("**Valor de Parcela**", min_value=limits["ValorQuitacao"][0], max_value=limits["ValorQuitacao"][1], value=(limits["ValorQuitacao"][0] + limits["ValorQuitacao"][1]) / 2, step=0.01)
atraso = st.slider("**Nº de Parcelas**", min_value=limits["Atraso"][0], max_value=limits["Atraso"][1], value=(limits["Atraso"][0] + limits["Atraso"][1]) / 2, step=1.0)
quant_pagamentos_via_boleto = st.slider("**Quant. de Boletos pagos**", min_value=limits["Quant_Pagamentos_Via_Boleto"][0], max_value=limits["Quant_Pagamentos_Via_Boleto"][1], value=(limits["Quant_Pagamentos_Via_Boleto"][0] + limits["Quant_Pagamentos_Via_Boleto"][1]) // 2, step=1)
quant_ocorrencia = st.slider("**Idade do Cliente**", min_value=limits["Quant_Ocorrencia"][0], max_value=limits["Quant_Ocorrencia"][1], value=(limits["Quant_Ocorrencia"][0] + limits["Quant_Ocorrencia"][1]) // 2, step=1)
uf = st.selectbox("**UF**", list(encoder.categories_[0]))

# Criar um DataFrame com os inputs do usuário
user_data = pd.DataFrame({
    "ValorQuitacao": [valor_quitacao],
    "Atraso": [atraso],
    "Quant_Pagamentos_Via_Boleto": [quant_pagamentos_via_boleto],
    "Quant_Ocorrencia": [quant_ocorrencia],
    "UF": [uf]
})

# Codificar a variável UF
uf_encoded_user = encoder.transform(user_data[["UF"]])
uf_encoded_user_df = pd.DataFrame(uf_encoded_user, columns=encoder.get_feature_names_out(["UF"]))

# Remover a coluna UF original e adicionar as novas colunas codificadas
user_data = user_data.drop(columns=["UF"]).reset_index(drop=True)
uf_encoded_user_df = uf_encoded_user_df.reset_index(drop=True)
user_data = pd.concat([user_data, uf_encoded_user_df], axis=1)

# Garantir que os nomes das colunas estejam corretos para previsão
user_data = user_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Fazer a previsão automaticamente ao alterar qualquer entrada
probabilidade = model.predict_proba(user_data)[:, 1][0]  # Probabilidade de ser 'Pago'
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
