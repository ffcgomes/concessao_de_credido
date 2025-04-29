import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def safe_onehot_transform(encoder, data, column_name):
    """
    Função ultra-robusta para transformação one-hot que:
    1. Lida com qualquer versão do scikit-learn
    2. Resolve problemas de shape mismatch
    3. Garante nomes de colunas consistentes
    """
    try:
        # Transformação dos dados
        encoded_data = encoder.transform(data[[column_name]])
        
        # Converter para array denso se for sparse
        if hasattr(encoded_data, 'toarray'):
            encoded_data = encoded_data.toarray()
        
        # Determinar o número correto de categorias
        if hasattr(encoder, 'categories_'):
            num_categories = len(encoder.categories_[0])
        else:
            num_categories = encoded_data.shape[1]
        
        # Ajustar o shape se necessário
        if encoded_data.shape[1] > num_categories:
            encoded_data = encoded_data[:, :num_categories]
        elif encoded_data.shape[1] < num_categories:
            encoded_data = np.pad(encoded_data, 
                                ((0, 0), (0, num_categories - encoded_data.shape[1])), 
                                mode='constant')
        
        # Gerar nomes de colunas compatíveis
        if hasattr(encoder, 'categories_'):
            col_names = [f"{column_name}_{cat}" for cat in encoder.categories_[0]]
        else:
            col_names = [f"{column_name}_{i}" for i in range(num_categories)]
        
        return pd.DataFrame(encoded_data, columns=col_names)
    
    except Exception as e:
        st.error(f"Erro na transformação: {str(e)}")
        st.stop()

@st.cache_data
def load_model():
    try:
        with open("./resultados_parciais/modelo_logistico.pkl", "rb") as f:
            loaded = joblib.load(f)
            return (loaded[0], loaded[1]) if isinstance(loaded, tuple) else (loaded, None)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        st.stop()

# Inicialização
model, encoder = load_model()

if encoder is None:
    st.error("Encoder não encontrado no arquivo do modelo")
    st.stop()

# Interface do usuário
st.title("🔍 Análise de Crédito Automatizada")

# Configurações
params = {
    "ValorParcela": {"min": 100.0, "max": 10000.0, "default": 2500.0},
    "ParcelasAtraso": {"min": 0, "max": 36, "default": 6},
    "BoletosPagos": {"min": 0, "max": 24, "default": 8},
    "IdadeCliente": {"min": 18, "max": 80, "default": 35}
}

# Formulário
with st.form("credit_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        valor = st.slider("Valor da Parcela (R$)", 
                         params["ValorParcela"]["min"],
                         params["ValorParcela"]["max"],
                         params["ValorParcela"]["default"])
        
        atraso = st.slider("Parcelas em Atraso", 
                          params["ParcelasAtraso"]["min"],
                          params["ParcelasAtraso"]["max"],
                          params["ParcelasAtraso"]["default"])
    
    with col2:
        boletos = st.slider("Boletos Pagos (últimos 6 meses)", 
                           params["BoletosPagos"]["min"],
                           params["BoletosPagos"]["max"],
                           params["BoletosPagos"]["default"])
        
        idade = st.slider("Idade do Cliente", 
                         params["IdadeCliente"]["min"],
                         params["IdadeCliente"]["max"],
                         params["IdadeCliente"]["default"])
    
    # Opções de UF dinâmicas
    uf_options = (encoder.categories_[0] if hasattr(encoder, 'categories_') 
                 else ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF"])
    
    uf = st.selectbox("UF de Residência", options=uf_options)
    
    submit = st.form_submit_button("Realizar Análise")

if submit:
    try:
        # Preparar dados de entrada
        input_df = pd.DataFrame({
            "ValorQuitacao": [valor],
            "Atraso": [atraso],
            "Quant_Pagamentos_Via_Boleto": [boletos],
            "Quant_Ocorrencia": [idade],
            "UF": [uf]
        })
        
        # Transformação segura
        encoded_uf = safe_onehot_transform(encoder, input_df, "UF")
        
        # Debug (opcional)
        st.write(f"Shape após encoding: {encoded_uf.shape}")
        
        # Combinar dados
        processed_data = pd.concat([
            input_df.drop(columns=["UF"]),
            encoded_uf
        ], axis=1)
        
        # Ajuste final para o modelo
        if hasattr(model, 'feature_names_in_'):
            # Adicionar colunas faltantes
            for col in model.feature_names_in_:
                if col not in processed_data.columns:
                    processed_data[col] = 0
            
            # Ordenar colunas
            processed_data = processed_data[model.feature_names_in_]
        
        # Predição
        prob = model.predict_proba(processed_data)[0, 1]
        
        # Exibir resultados
        st.subheader("Resultado da Análise")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Probabilidade de Pagamento", 
                    f"{prob:.1%}",
                    help="Probabilidade estimada de o cliente honrar com o pagamento")
            
        with col_res2:
            st.metric("Recomendação", 
                    "Aprovar" if prob > 0.5 else "Rejeitar",
                    delta="✅ Favorável" if prob > 0.5 else "❌ Desfavorável",
                    delta_color="normal")
        
        # Barra de progresso visual
        st.progress(prob)
        
    except Exception as e:
        st.error(f"Erro durante a análise: {str(e)}")

st.markdown("---")
st.caption("Sistema de análise preditiva v3.0 | © 2023")