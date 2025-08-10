# app.py (v3.6 - Painel de Estratégias)

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

# ... (Seções de Gerenciamento de Estratégias e Configurações Financeiras permanecem as mesmas) ...
# ==============================================================================
# --- GERENCIAMENTO DE ESTRATÉGIAS (Carregar e Salvar) ---
# ==============================================================================
ESTRATEGIAS_FILE = "estrategias.json"

# Valores que vamos calibrar
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
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return [ESTRATEGIA_PADRAO_DEFAULT, ESTRATEGIA_CONTRARIO_DEFAULT]
    return [ESTRATEGIA_PADRAO_DEFAULT, ESTRATEGIA_CONTRARIO_DEFAULT]

def salvar_estrategias(lista_estrategias):
    with open(ESTRATEGIAS_FILE, 'w') as f:
        json.dump(lista_estrategias, f, indent=4)
# ==============================================================================
# --- CONFIGURAÇÕES FINANCEIRAS ---
# ==============================================================================
CUSTO_JOGO_15_DEZENAS = 3.50
PREMIOS_FIXOS = { 11: 7.0, 12: 14.0, 13: 35.0 }

# ==============================================================================
# --- CLASSE DO ANALISADOR ---
# ==============================================================================
class LotofacilAnalisador:
    def __init__(self, df, config, model=None):
        self.df = df; self.config = config; self.model = model
        self.universo = list(range(1, 26)); self.miolo = {7, 8, 9, 12, 13, 14, 17, 18, 19}
        self.estrategias = config.get("ESTRATEGIAS_ATUAIS", [])
        self.search_spaces = {
            "Descoberto": [
                Integer(0, 100, name='peso_proximidade_soma'),
                Integer(0, 100, name='peso_proximidade_pares'),
                Integer(0, 100, name='peso_proximidade_repetidas'),
                Integer(0, 50, name='peso_por_fria'),
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
        stats = {'soma': sum(jogo), 'pares': sum(1 for n in jogo if n % 2 == 0), 'repetidas': len(set(jogo) & set(dados['concurso_anterior'])), 'frias': len(set(jogo) & set(dados['frias'])), 'quentes': len(set(jogo) & set(dados['quentes']))}
        if alvo:
            dist_soma = abs(stats['soma'] - alvo['Soma Média']); dist_pares = abs(stats['pares'] - alvo['Pares Média']); dist_repetidas = abs(stats['repetidas'] - alvo['Repetidas Média'])
            score += (1 / (1 + dist_soma)) * pesos.get('peso_proximidade_soma', 0)
            score += (1 / (1 + dist_pares)) * pesos.get('peso_proximidade_pares', 0)
            score += (1 / (1 + dist_repetidas)) * pesos.get('peso_proximidade_repetidas', 0)
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

    # ... (os métodos _obter_analise_ia, gerar_jogos_com_analise e os de ML permanecem os mesmos) ...
    def _obter_analise_ia(self, jogos_com_score, dados, regras, perfil):
        # ... (código inalterado)
    def gerar_jogos_com_analise(self, estrategia, n_jogos):
        # ... (código inalterado)
    def descobrir_perfis(self, n_perfis):
        # ... (código inalterado)
    def _backtest_silencioso(self, pesos, alvo):
        # ... (código inalterado)
    def otimizar_estrategia(self, perfil_alvo, status_placeholder):
        # ... (código inalterado)

    # --- NOVO MÉTODO PARA SIMULAR UMA ÚNICA ESTRATÉGIA ---
    def simular_estrategia_unica(self, estrategia):
        concursos_para_testar = self.df.tail(self.config['NUMERO_DE_CONCURSOS_A_TESTAR'])
        resultados = {'resumo_acertos': {i: 0 for i in range(11, 16)}, 'custo': 0.0, 'retorno': 0.0}
        historico_lucro_acumulado = []; lucro_total_acumulado = 0.0
        
        barra_progresso = st.progress(0, text=f"Simulando estratégia '{estrategia['nome']}'...")

        for i, (_, concurso_alvo) in enumerate(concursos_para_testar.iterrows()):
            num_concurso_alvo = int(concurso_alvo['Concurso'])
            df_historico = self.df[self.df['Concurso'] < num_concurso_alvo]
            if df_historico.empty: continue
            
            resultado_real = sorted([int(concurso_alvo[f'Bola{i}']) for i in range(1, 16)])
            dados = {'concurso_anterior': sorted([int(df_historico.iloc[-1][f'Bola{i}']) for i in range(1, 16)]), 'quentes': [], 'frias': self._get_dezenas_frias(df_historico, n=8)}
            
            jogos_gerados = self._gerar_melhores_jogos(self.config['JOGOS_POR_SIMULACAO'], dados, estrategia)
            
            custo_neste_concurso = 0.0
            retorno_neste_concurso = 0.0
            for jogo, score in jogos_gerados:
                acertos = len(set(jogo) & set(resultado_real))
                if 11 <= acertos <= 15:
                    resultados['resumo_acertos'][acertos] += 1
                premio = PREMIOS_FIXOS.get(acertos, 0.0)
                resultados['custo'] += CUSTO_JOGO_15_DEZENAS
                custo_neste_concurso += CUSTO_JOGO_15_DEZENAS
                if premio > 0:
                    resultados['retorno'] += premio
                    retorno_neste_concurso += premio

            lucro_neste_concurso = retorno_neste_concurso - custo_neste_concurso
            lucro_total_acumulado += lucro_neste_concurso
            historico_lucro_acumulado.append(lucro_total_acumulado)
            barra_progresso.progress((i + 1) / len(concursos_para_testar))

        barra_progresso.empty()
        return resultados, historico_lucro_acumulado

# ==============================================================================
# --- FUNÇÕES DE APOIO E INTERFACE ---
# ==============================================================================
@st.cache_data(ttl=3600)
def carregar_dados():
    # ... (código inalterado)
@st.cache_resource
def carregar_modelo_ia():
    # ... (código inalterado)

# ==============================================================================
# --- INTERFACE PRINCIPAL DO STREAMLIT ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Analisador Lotofácil com IA")
st.title("🤖 Analisador Inteligente da Lotofácil")

st.sidebar.title("Ferramentas")
modo_app = st.sidebar.radio(
    "Escolha o que deseja fazer:",
    ["Gerador de Jogos", "Painel de Estratégias", "Laboratório de IA"],
    key="navigation"
)

try:
    df = carregar_dados()
    model = carregar_modelo_ia() if modo_app == "Gerador de Jogos" else None
    st.sidebar.success(f"Base de dados carregada! \n\nÚltimo concurso: {df['Concurso'].max()}")
    
    with st.sidebar.expander("Ver Estratégias Salvas", expanded=True):
        estrategias_atuais = carregar_estrategias()
        if not estrategias_atuais:
            st.write("Nenhuma estratégia salva. Use o Laboratório de IA.")
        for i, est in enumerate(estrategias_atuais):
            st.write(f"**{est['nome']}** (ROI Salvo: {est['roi']:.2f}%)")

except Exception as e:
    st.error(f"Ocorreu um erro fatal no carregamento: {e}"); st.stop()

# ... (Página do Gerador de Jogos permanece a mesma) ...

# --- NOVA PÁGINA: PAINEL DE ESTRATÉGIAS ---
elif modo_app == "Painel de Estratégias":
    st.header("📊 Painel de Controle de Estratégias")
    st.info("Aqui você pode gerenciar, analisar e comparar o desempenho histórico de todas as suas estratégias salvas.")

    estrategias_salvas = carregar_estrategias()
    nomes_estrategias = [est['nome'] for est in estrategias_salvas]
    
    if not nomes_estrategias:
        st.warning("Nenhuma estratégia salva encontrada. Use o 'Laboratório de IA' para descobrir e salvar novas estratégias.")
    else:
        estrategia_selecionada_nome = st.selectbox("Selecione uma estratégia para analisar em detalhes:", nomes_estrategias)
        estrategia_obj = next((item for item in estrategias_salvas if item["nome"] == estrategia_selecionada_nome), None)
        
        if estrategia_obj:
            st.subheader(f"Análise da Estratégia: {estrategia_obj['nome']}")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Tipo:**", estrategia_obj['tipo'])
                st.write("**ROI Salvo:**", f"{estrategia_obj['roi']:.2f}%")
                st.write("**Alvo (Características Médias):**")
                st.json(estrategia_obj.get('alvo', 'N/A para estratégias manuais'))
                st.write("**Pesos da Estratégia:**")
                st.json(estrategia_obj['pesos'])
                
                if st.button("🚨 Excluir esta Estratégia"):
                    estrategias_salvas.remove(estrategia_obj)
                    salvar_estrategias(estrategias_salvas)
                    st.success(f"Estratégia '{estrategia_obj['nome']}' foi excluída. A página será recarregada.")
                    st.experimental_rerun()

            with col2:
                st.markdown("##### Simulação de Desempenho Histórico")
                st.warning("A simulação abaixo é uma estimativa baseada nos últimos 100 concursos e pode variar.")
                
                if st.button("Rodar Simulação de Desempenho"):
                    CONFIG = {"NUMERO_DE_CONCURSOS_A_TESTAR": 100, "JOGOS_POR_SIMULACAO": 10, "NUM_CANDIDATOS": 5000}
                    analisador = LotofacilAnalisador(df, CONFIG)
                    
                    with st.spinner("Realizando backtest..."):
                        resultados, historico_lucro = analisador.simular_estrategia_unica(estrategia_obj)
                    
                    lucro = resultados['retorno'] - resultados['custo']
                    roi = (lucro / resultados['custo'] * 100) if resultados['custo'] > 0 else 0
                    
                    st.metric(f"Resultado da Simulação (100 Concursos)", f"R$ {lucro:.2f}", f"{roi:.2f}% ROI")

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(historico_lucro, marker='o', linestyle='-', color='royalblue')
                    ax.axhline(0, color='red', linestyle='--', linewidth=1.2, label='Ponto de Equilíbrio (R$ 0)')
                    ax.set_title(f'Evolução do Saldo para a Estratégia "{estrategia_obj["nome"]}"'); ax.set_xlabel('Concursos Simulados'); ax.set_ylabel('Saldo Acumulado (R$)')
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.legend()
                    st.pyplot(fig)

# ... (Página do Laboratório de IA permanece a mesma) ...
