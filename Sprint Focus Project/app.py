"""
app_outliers_iqr.py

Aplicativo Streamlit: Limpeza de Outliers com IQR

Este app reutiliza apenas conceitos básicos do Streamlit:
- estrutura de página
- sidebar
- session_state
- widgets
- layout em colunas
- visualização
- ação explícita
- exportação

O objetivo é transformar um procedimento comum de dados
em uma interface simples, interpretável e reutilizável.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# 1) Configuração básica da página
# ============================================================
# Define o contexto visual do app.
# Não altera a lógica, apenas a apresentação.
st.set_page_config(
    page_title="IQR Outlier Cleaner",
    page_icon="🧼",
    layout="wide"
)

st.title("🧼 Limpeza de Outliers com IQR")
st.caption("Preview → decisão → aplicação")


# ============================================================
# 2) Memória do app (session_state)
# ============================================================
# Como o Streamlit reexecuta o script a cada interação,
# usamos session_state para manter dados entre ações.

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None   # dataset original

if "df" not in st.session_state:
    st.session_state.df = None       # dataset em uso

if "status" not in st.session_state:
    st.session_state.status = "Carregue um arquivo para começar."


# ============================================================
# 3) Sidebar: entrada de dados e controle global
# ============================================================
st.sidebar.title("Entrada de dados")

uploaded = st.sidebar.file_uploader(
    "Envie um CSV",
    type=["csv"]
)

# Quando o usuário envia um arquivo:
# - lemos o CSV
# - guardamos uma cópia original
# - criamos uma cópia de trabalho
if uploaded is not None:
    df0 = pd.read_csv(uploaded)

    st.session_state.raw_df = df0.copy(deep=True)
    st.session_state.df = df0.copy(deep=True)

    st.session_state.status = (
        f"Arquivo carregado | "
        f"{df0.shape[0]} linhas | {df0.shape[1]} colunas"
    )

# Feedback de status sempre visível
st.sidebar.info(st.session_state.status)

# Botão de reset:
# devolve o dataset ao estado original
if st.sidebar.button(
    "Reset",
    use_container_width=True,
    disabled=(st.session_state.raw_df is None)
):
    st.session_state.df = st.session_state.raw_df.copy(deep=True)
    st.session_state.status = "Reset aplicado. Dataset original restaurado."


# ============================================================
# 4) Bloqueio de fluxo (prevenção de erro)
# ============================================================
# Se não houver dados carregados, o app não continua.
if st.session_state.df is None:
    st.warning("Use a sidebar para carregar um arquivo CSV.")
    st.stop()

df = st.session_state.df


# ============================================================
# 5) Função de IQR clipping
# ============================================================
def iqr_clip_df(df: pd.DataFrame, cols: list[str], k: float):
    """
    Aplica IQR clipping em colunas numéricas.

    Para cada coluna:
    - calcula Q1 e Q3
    - calcula IQR = Q3 - Q1
    - define limites inferior e superior
    - corta valores fora do intervalo

    Retorna:
    - novo DataFrame
    - relatório com impacto da transformação
    """
    df2 = df.copy()
    rows = []

    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)

        df2[col] = df[col].clip(lo, hi)

        rows.append({
            "coluna": col,
            "valores_afetados": ((df[col] < lo) | (df[col] > hi)).sum()
        })

    return df2, pd.DataFrame(rows)


# ============================================================
# 6) Configuração dos parâmetros (widgets)
# ============================================================
st.subheader("Configuração")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not num_cols:
    st.error("O dataset não possui colunas numéricas.")
    st.stop()

cols_sel = st.multiselect(
    "Colunas numéricas para aplicar IQR",
    options=num_cols,
    default=num_cols[: min(3, len(num_cols))]
)

k = st.slider(
    "Fator k (agressividade do corte)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.25
)

preview_col = st.selectbox(
    "Coluna para visualização",
    options=(cols_sel if cols_sel else num_cols)
)


# ============================================================
# 7) Preview antes de aplicar
# ============================================================
st.markdown("---")
st.subheader("Preview: antes vs depois")

if not cols_sel:
    st.info("Selecione ao menos uma coluna.")
    st.stop()

df_after, report = iqr_clip_df(df, cols_sel, k)


fig, ax = plt.subplots()
ax.hist(df[preview_col].dropna(), bins=40, alpha=0.6, label="Antes")
ax.hist(df_after[preview_col].dropna(), bins=40, alpha=0.6, label="Depois")
ax.set_title(f"Histograma: {preview_col}")
ax.legend()
st.pyplot(fig)


# ============================================================
# 8) Aplicar transformação (ação explícita)
# ============================================================
if st.button("Aplicar no dataset", type="primary"):
    st.session_state.df = df_after
    st.session_state.status = "IQR clipping aplicado com sucesso."


