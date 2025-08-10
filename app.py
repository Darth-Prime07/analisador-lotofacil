# app.py (v3.9 - Conferidor de Apostas)

import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import statistics
import gspread
import google.generativeai as genai
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re

# ==============================================================================
# --- GERENCIAMENTO DE ESTRATÉGIAS E CONFIGS ---
# (Esta seção permanece a mesma)
# ==============================================================================
ESTRATEGIAS_FILE = "estrategias.json"
ESTRATEGIA_PADRAO_DEFAULT = {
    "nome": "Padrão (Manual)", "tipo": "Manual", "roi": -70.0,
    "alvo": {'Soma Média': 195, 'Pares Média': 8, 'Repetidas Média': 9, 'Primos Média': 5, 'Moldura Média': 10},
    "pesos": {'peso_proximidade_soma': 20, 'peso_proximidade_pares': 10, 'peso_proximidade_repetidas': 20, 'peso_por_fria': 15, 'peso_por_quente': 5, 'peso_proximidade_primos': 5, 'peso_proximidade_moldura': 5}
}
ESTRATEGIA_CONTRARIO_DEFAULT = {
    "nome": "Contrário (Manual)", "tipo": "Manual", "roi": -70.0,
    "alvo": {'Soma Média': 180, 'Pares Média': 7, 'Repetidas Média': 7, 'Primos Média': 6, 'Moldura Média': 11},
    "pesos": {'peso_proximidade_soma': 10, 'peso_proximidade_pares': 5, 'peso_proximidade_repetidas': 10, 'peso_por_fria': 25, 'peso_por_quente': -10, 'peso_proximidade_primos': 0, 'peso_proximidade_moldura': 0}
}
def carregar_estrategias():
    if os.path.exists(ESTRATEGIAS_FILE):
        with open(ESTRATEGIAS_FILE, 'r') as f:
            try: return json.load(f)
            except json.JSONDecodeError: return [ESTRATEGIA_PADRAO_DEFAULT, ESTRATEGIA_CONTRARIO_DEFAULT]
    return [ESTRATEGIA_PADRAO_DEFAULT, ESTRATEGIA_CONTRARIO_DEFAULT]
def salvar_estrategias(lista_estrategias):
    with open(ESTRATEGIAS_FILE, 'w') as f:
        json.dump(lista_estrategias, f, indent=4)
CUSTO_JOGO_15_DEZENAS = 3.50
PREMIOS_FIXOS = { 11: 7.0, 12: 14.0, 13: 35.0 }

# ==============================================================================
# --- CLASSE DO ANALISADOR ---
# (A classe inteira permanece a mesma)
# ==============================================================================
class LotofacilAnalisador:
    def __init__(self, df, config, model=None):
        self.df = df; self.config = config; self.model = model
        self.universo = list(range(1, 26)); self.miolo = {7, 8, 9, 12, 13, 14, 17, 18, 19}
        self.primos = {2, 3, 5, 7, 11, 13, 17, 19, 23}
        self.moldura = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
        self.estrategias = config.get("ESTRATEGIAS_ATUAIS", [])
        self.search_spaces = {
            "Descoberto": [
                Integer(0, 100, name='peso_proximidade_soma'), Integer(0, 100, name='peso_proximidade_pares'),
                Integer(0, 100, name='peso_proximidade_repetidas'), Integer(0, 50, name='peso_proximidade_primos'),
                Integer(0, 50, name='peso_proximidade_moldura'), Integer(0, 50, name='peso_por_fria'),
                Integer(-50, 0, name='peso_por_quente')
            ]
        }
    def _get_dezenas_frias(self, df, n=8):
        dezenas = np.arange(1, 26); atraso = {}
        ultimo_concurso_geral = df['Concurso'].max(); colunas_dezenas = [f'Bola{i+1}' for i in range(15)]
        for dezena in dezenas:
            ultimo_concurso_dezena = df[df[colunas_dezenas].eq(dezena).any(axis=1)]['Concurso'].max()
            atraso[dezena] = len(df) if pd.isna(ultimo_concurso_dezena) else ultimo_concurso_geral - ultimo_concurso_dezena
        return sorted(atraso, key=atraso.get, reverse=True)[:n]
    def _analisar_tendencias(self, df_historico, n_concursos=15):
        df_tendencia = df_historico.tail(n_concursos); regras = {}
        colunas_bolas = [f'Bola{i+1}' for i in range(15)]
        repetidas = [len(set(df_tendencia.iloc[i+1][colunas_bolas]) & set(df_tendencia.iloc[i][colunas_bolas])) for i in range(len(df_tendencia) - 1)]
        regras['repetidas_ideal'] = statistics.mode(repetidas) if repetidas else 9
        somas = [sum(row[colunas_bolas]) for _, row in df_tendencia.iterrows()]
        regras['soma_ideal'] = int(statistics.mean(somas)) if somas else 195
        return regras
    def _gerar_candidato_guiado(self, dezenas_a_gerar):
        return sorted(random.sample(self.universo, dezenas_a_gerar))
    def _pontuar_jogo_dinamico(self, jogo, dados, pesos, alvo):
        score = 0
        stats = {
            'soma': sum(jogo), 'pares': sum(1 for n in jogo if n % 2 == 0),
            'repetidas': len(set(jogo) & set(dados['concurso_anterior'])),
            'frias': len(set(jogo) & set(dados['frias'])), 'quentes': len(set(jogo) & set(dados['quentes'])),
            'primos': len(set(jogo) & self.primos), 'moldura': len(set(jogo) & self.moldura)
        }
        if alvo:
            dist_soma = abs(stats['soma'] - alvo['Soma Média'])
            dist_pares = abs(stats['pares'] - alvo['Pares Média'])
            dist_repetidas = abs(stats['repetidas'] - alvo['Repetidas Média'])
            dist_primos = abs(stats['primos'] - alvo.get('Primos Média', 5))
            dist_moldura = abs(stats['moldura'] - alvo.get('Moldura Média', 10))
            score += (1 / (1 + dist_soma)) * pesos.get('peso_proximidade_soma', 0)
            score += (1 / (1 + dist_pares)) * pesos.get('peso_proximidade_pares', 0)
            score += (1 / (1 + dist_repetidas)) * pesos.get('peso_proximidade_repetidas', 0)
            score += (1 / (1 + dist_primos)) * pesos.get('peso_proximidade_primos', 0)
            score += (1 / (1 + dist_moldura)) * pesos.get('peso_proximidade_moldura', 0)
        score += stats['frias'] * pesos.get('peso_por_fria', 0)
        score += stats['quentes'] * pesos.get('peso_por_quente', 0)
        return score
    def _gerar_melhores_jogos(self, n_jogos, dados, estrategia):
        pesos = estrategia['pesos']; alvo = estrategia.get('alvo'); num_candidatos = self.config.get("NUM_CANDIDATOS", 10000)
        jogos_pontuados = []
        for _ in range(num_candidatos):
            jogo = self._gerar_candidato_guiado(15)
            score = self._pontuar_jogo_dinamico(jogo, dados, pesos, alvo)
            jogos_pontuados.append((jogo, score))
        jogos_pontuados.sort(key=lambda x: x[1], reverse=True)
        jogos_finais_com_score = []
        jogos_vistos = set()
        for jogo, score in jogos_pontuados:
            if tuple(jogo) not in jogos_vistos:
                jogos_finais_com_score.append((jogo, score))
                jogos_vistos.add(tuple(jogo))
            if len(jogos_finais_com_score) == n_jogos: break
        return jogos_finais_com_score
    def gerar_jogos_com_analise(self, estrategia, n_jogos):
        st.toast("Analisando tendências do último concurso...")
        regras_dinamicas = self._analisar_tendencias(self.df)
        dados_futuro = {}; dados_futuro['concurso_anterior'] = sorted([int(self.df.iloc[-1][f'Bola{i}']) for i in range(1, 16)])
        frequencias = pd.Series(self.df[[f'Bola{i}' for i in range(1, 16)]].values.flatten()).value_counts()
        dados_futuro['quentes'] = sorted(list(frequencias.head(8).index.astype(int)))
        dados_futuro['frias'] = self._get_dezenas_frias(self.df, n=8)
        st.toast(f"Gerando e pontuando jogos com a estratégia '{estrategia['nome']}'...")
        jogos_com_score = self._gerar_melhores_jogos(n_jogos, dados_futuro, estrategia)
        st.toast("Solicitando análise da IA do Gemini...")
        analise_ia = self._obter_analise_ia(jogos_com_score, dados_futuro, regras_dinamicas, estrategia['nome'])
        return {"jogos": jogos_com_score, "analise": analise_ia}
    def descobrir_perfis(self, n_perfis):
        # ... (código inalterado)
    def _backtest_silencioso(self, pesos, alvo):
        # ... (código inalterado)
    def otimizar_estrategia(self, perfil_alvo, status_placeholder):
        # ... (código inalterado)
    def simular_estrategia_unica(self, estrategia):
        # ... (código inalterado)
    def rodar_simulacao(self):
        # ... (código inalterado)
    def _obter_analise_ia(self, jogos_com_score, dados, regras, perfil):
        # ... (código inalterado)

# ==============================================================================
# --- FUNÇÕES DE APOIO E INTERFACE ---
# ==============================================================================
@st.cache_data(ttl=3600)
def carregar_dados():
    # ... (código inalterado)
@st.cache_resource
def carregar_modelo_ia():
    # ... (código inalterado)

# --- NOVA FUNÇÃO PARA O CONFERIDOR ---
def conferir_jogos(texto_resultado, texto_apostas):
    # Limpa e converte a string do resultado para um set de inteiros
    numeros_resultado = set(map(int, re.findall(r'\d+', texto_resultado)))
    if len(numeros_resultado) != 15:
        st.error(f"O resultado oficial deve ter 15 dezenas. Você inseriu {len(numeros_resultado)}.")
        return None

    # Limpa e converte a string de apostas para uma lista de listas de inteiros
    apostas = []
    linhas = texto_apostas.strip().split('\n')
    for i, linha in enumerate(linhas):
        try:
            aposta = list(map(int, re.findall(r'\d+', linha)))
            if len(aposta) == 15:
                apostas.append(aposta)
            else:
                st.warning(f"A linha {i+1} dos seus jogos foi ignorada pois não continha 15 dezenas.")
        except:
            st.warning(f"A linha {i+1} dos seus jogos foi ignorada por um erro de formato.")
    
    if not apostas:
        st.error("Nenhum jogo válido com 15 dezenas foi encontrado no campo 'Meus Jogos'.")
        return None

    # Calcula os resultados
    resultados_detalhados = []
    resumo_acertos = {i: 0 for i in range(11, 16)}
    retorno_total = 0.0

    for aposta in apostas:
        acertos = len(set(aposta) & numeros_resultado)
        premio = PREMIOS_FIXOS.get(acertos, 0.0)
        retorno_total += premio
        if acertos >= 11:
            resumo_acertos[acertos] += 1
        resultados_detalhados.append({
            "Jogo": ", ".join(map(str, sorted(aposta))),
            "Acertos": acertos,
            "Prêmio (R$)": f"{premio:.2f}"
        })
    
    custo_total = len(apostas) * CUSTO_JOGO_15_DEZENAS
    lucro = retorno_total - custo_total
    
    return {
        "custo": custo_total, "retorno": retorno_total, "lucro": lucro,
        "resumo_acertos": resumo_acertos, "detalhes": pd.DataFrame(resultados_detalhados)
    }

# ==============================================================================
# --- INTERFACE PRINCIPAL DO STREAMLIT ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Analisador Lotofácil com IA")
st.title("🤖 Analisador Inteligente da Lotofácil")

st.sidebar.title("Ferramentas")
modo_app = st.sidebar.radio(
    "Escolha o que deseja fazer:",
    ["Gerador de Jogos", "Conferidor de Apostas", "Painel de Estratégias", "Laboratório de IA"],
    key="navigation"
)
# ... (código de carregamento de dados e sidebar expander inalterado) ...

# --- PÁGINA DO GERADOR ---
if modo_app == "Gerador de Jogos":
    # ... (código inalterado)

# --- NOVA PÁGINA: CONFERIDOR DE APOSTAS ---
elif modo_app == "Conferidor de Apostas":
    st.header("🎯 Conferidor de Apostas")
    st.info("Cole o resultado oficial e os jogos que você apostou para ver seu desempenho.")

    col1, col2 = st.columns(2)

    with col1:
        texto_resultado = st.text_area(
            "Cole aqui o resultado oficial (15 dezenas)",
            placeholder="Ex: 1, 2, 3, 5, 8, 9, 10, 12, 14, 15, 17, 20, 21, 23, 25"
        )
    with col2:
        texto_apostas = st.text_area(
            "Cole aqui os seus jogos (um por linha)",
            placeholder="[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]\nJogo 2 (Padrão): [1, 2, 3, 6, 10, 11, 12, 14...]",
            height=200
        )

    if st.button("Conferir Resultados", type="primary", use_container_width=True):
        if not texto_resultado or not texto_apostas:
            st.warning("Por favor, preencha ambos os campos.")
        else:
            resultados = conferir_jogos(texto_resultado, texto_apostas)

            if resultados:
                st.subheader("Relatório de Desempenho")
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Lucro / Prejuízo", f"R$ {resultados['lucro']:.2f}")
                kpi2.metric("Custo Total", f"R$ {resultados['custo']:.2f}")
                kpi3.metric("Retorno Total", f"R$ {resultados['retorno']:.2f}")

                st.markdown("---")
                
                st.dataframe(resultados['detalhes'], use_container_width=True)

# --- PÁGINA DO PAINEL DE ESTRATÉGIAS ---
elif modo_app == "Painel de Estratégias":
    # ... (código inalterado)

# --- PÁGINA DO LABORATÓRIO DE IA ---
elif modo_app == "Laboratório de IA":
    # ... (código inalterado)
