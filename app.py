# app.py (v3.8 - Descobridor Enriquecido com Novas Features)

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

# ==============================================================================
# --- GERENCIAMENTO DE ESTRATÉGIAS, CONFIGS E CLASSE ---
# (Nenhuma mudança nesta parte, todo o código é o mesmo da v3.7)
# ==============================================================================
ESTRATEGIAS_FILE = "estrategias.json"
ESTRATEGIA_PADRAO_DEFAULT = {
    "nome": "Padrão (Manual)", "tipo": "Manual", "roi": -70.0,
    "alvo": {'Soma Média': 195, 'Pares Média': 8, 'Repetidas Média': 9},
    "pesos": {'peso_proximidade_soma': 20, 'peso_proximidade_pares': 10, 'peso_proximidade_repetidas': 20, 'peso_por_fria': 15, 'peso_por_quente': 5}
}
ESTRATEGIA_CONTRARIO_DEFAULT = {
    "nome": "Contrário (Manual)", "tipo": "Manual", "roi": -70.0,
    "alvo": {'Soma Média': 180, 'Pares Média': 7, 'Repetidas Média': 7},
    "pesos": {'peso_proximidade_soma': 10, 'peso_proximidade_pares': 5, 'peso_proximidade_repetidas': 10, 'peso_por_fria': 25, 'peso_por_quente': -10}
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
                Integer(0, 100, name='peso_proximidade_repetidas'), Integer(0, 50, name='peso_por_fria'),
                Integer(-50, 0, name='peso_por_quente'), Integer(0, 50, name='peso_proximidade_primos'),
                Integer(0, 50, name='peso_proximidade_moldura')
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
            dist_primos = abs(stats['primos'] - alvo['Primos Média'])
            dist_moldura = abs(stats['moldura'] - alvo['Moldura Média'])
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

    # --- MÉTODO DE DESCOBERTA ENRIQUECIDO ---
    def descobrir_perfis(self, n_perfis):
        df_analise = self.df.copy()
        bola_cols = [f'Bola{i}' for i in range(1, 16)]
        
        # 1. Feature Engineering (Cálculo das características para todo o histórico)
        df_anterior = self.df[bola_cols].shift(1)
        def calcular_repetidas(row):
            if pd.isna(row.iloc[-15:].sum()): return None
            concurso_atual = set(row.iloc[:15].values)
            concurso_anterior = set(row.iloc[-15:].values)
            return len(concurso_atual & concurso_anterior)

        df_merged = pd.concat([df_analise[bola_cols], df_anterior.rename(columns=lambda c: f"{c}_ant")], axis=1)
        
        df_analise['soma'] = df_analise[bola_cols].sum(axis=1)
        df_analise['pares'] = df_analise[bola_cols].apply(lambda row: sum(1 for n in row if n % 2 == 0), axis=1)
        df_analise['repetidas'] = df_merged.apply(calcular_repetidas, axis=1)
        df_analise['primos'] = df_analise[bola_cols].apply(lambda row: len(set(row) & self.primos), axis=1)
        df_analise['moldura'] = df_analise[bola_cols].apply(lambda row: len(set(row) & self.moldura), axis=1)
        
        features_list = ['soma', 'pares', 'repetidas', 'primos', 'moldura']
        features_df = df_analise[features_list].dropna()

        # 2. Normalizar os dados
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)

        # 3. Rodar o K-Means para encontrar os clusters
        kmeans = KMeans(n_clusters=n_perfis, random_state=42, n_init='auto')
        features_df['perfil_descoberto'] = kmeans.fit_predict(features_scaled)

        # 4. Analisar e retornar os centros dos clusters
        centros_reais = scaler.inverse_transform(kmeans.cluster_centers_)
        perfis_analisados = []
        for i, centro in enumerate(centros_reais):
            perfil = {
                "Soma Média": round(centro[0]), "Pares Média": round(centro[1]),
                "Repetidas Média": round(centro[2]), "Primos Média": round(centro[3]),
                "Moldura Média": round(centro[4]),
                "Ocorrências": int(features_df['perfil_descoberto'].value_counts()[i])
            }
            perfis_analisados.append(perfil)
        
        return perfis_analisados, features_df['perfil_descoberto'].value_counts()

    # (demais métodos como _backtest_silencioso, otimizar_estrategia, etc., permanecem os mesmos,
    # mas serão adaptados no futuro para usar as novas features nos pesos)
    def _backtest_silencioso(self, pesos, alvo):
        # ... (código inalterado)
    def otimizar_estrategia(self, perfil_alvo, status_placeholder):
        # ... (código inalterado)
    def gerar_jogos_com_analise(self, estrategia, n_jogos):
        # ... (código inalterado)
    def simular_estrategia_unica(self, estrategia):
        # ... (código inalterado)
        
# ==============================================================================
# --- FUNÇÕES DE APOIO E INTERFACE ---
# ==============================================================================
# (Nenhuma alteração aqui, todo o código é o mesmo da v3.7)
@st.cache_data(ttl=3600)
def carregar_dados():
    # ... (código inalterado)
@st.cache_resource
def carregar_modelo_ia():
    # ... (código inalterado)
def gerar_graficos(resumo_financeiro, historico_lucro):
    # ... (código inalterado)

# ==============================================================================
# --- INTERFACE PRINCIPAL DO STREAMLIT ---
# ==============================================================================
# (Pequenos ajustes na página do Laboratório de IA para exibir os novos perfis)

st.set_page_config(layout="wide", page_title="Analisador Lotofácil com IA")
st.title("🤖 Analisador Inteligente da Lotofácil")

# ... (código da navegação e carregamento de dados inalterado) ...

# --- PÁGINA DO LABORATÓRIO DE IA (MODIFICADA) ---
if modo_app == "Laboratório de IA":
    st.header("🔬 Laboratório de IA: Descoberta e Otimização")
    st.info("Esta ferramenta usa Machine Learning para analisar todo o histórico de resultados e descobrir 'perfis' de sorteios que ocorrem com mais frequência.")
    st.warning("Este processo analisa milhares de concursos e pode levar um ou dois minutos.", icon="⏳")

    st.sidebar.header("Configurações da Descoberta")
    n_perfis = st.sidebar.number_input("Quantos perfis você quer que a IA procure?", min_value=2, max_value=10, value=5, step=1)
    
    if st.sidebar.button("Analisar Histórico e Descobrir Perfis", type="primary", use_container_width=True):
        CONFIG = {}
        analisador = LotofacilAnalisador(df, CONFIG)
        
        with st.spinner("Processando todo o histórico e aplicando o algoritmo de clustering..."):
            perfis_encontrados, contagem_perfis = analisador.descobrir_perfis(n_perfis)

        st.subheader(f"Análise Concluída: {len(perfis_encontrados)} Perfis de Jogo Descobertos")
        st.markdown("Abaixo estão as características médias de cada perfil que a IA encontrou nos dados históricos, agora com muito mais detalhes.")

        for perfil in perfis_encontrados:
            nome = f"Perfil Descoberto {perfis_encontrados.index(perfil) + 1}"
            st.markdown(f"#### {nome}")
            st.json(perfil)
        
        st.subheader("Distribuição de Ocorrências dos Perfis")
        # Renomeia os índices do gráfico para nomes mais amigáveis
        contagem_perfis.index = [f"Perfil {i+1}" for i in contagem_perfis.index]
        st.bar_chart(contagem_perfis)
        
        st.info("Próximo passo: Use os insights destes perfis para criar e otimizar novas estratégias na aba 'Otimizador'.")
        
    else:
        st.info("Ajuste o número de perfis a procurar e clique no botão para iniciar a descoberta.")

# (As outras páginas, Gerador e Painel de Estratégias, permanecem as mesmas por enquanto)
# ...
