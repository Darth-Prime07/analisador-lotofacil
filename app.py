# app.py (v3.5 - Versão Completa, Funcional e Corrigida)

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
# --- GERENCIAMENTO DE ESTRATÉGIAS (Carregar e Salvar) ---
# ==============================================================================
ESTRATEGIAS_FILE = "estrategias.json"

ESTRATEGIA_PADRAO_DEFAULT = {
    "nome": "Padrão (Manual)", "tipo": "Manual", "roi": -75.0,
    "alvo": {'Soma Média': 195, 'Pares Média': 8, 'Repetidas Média': 9},
    "pesos": {'peso_proximidade_soma': 20, 'peso_proximidade_pares': 10, 'peso_proximidade_repetidas': 20, 'peso_por_fria': 15, 'peso_por_quente': 5}
}
ESTRATEGIA_CONTRARIO_DEFAULT = {
    "nome": "Contrário (Manual)", "tipo": "Manual", "roi": -75.0,
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
        stats = {
            'soma': sum(jogo),
            'pares': sum(1 for n in jogo if n % 2 == 0),
            'repetidas': len(set(jogo) & set(dados['concurso_anterior'])),
            'frias': len(set(jogo) & set(dados['frias'])),
            'quentes': len(set(jogo) & set(dados['quentes']))
        }
        if alvo:
            dist_soma = abs(stats['soma'] - alvo['Soma Média'])
            dist_pares = abs(stats['pares'] - alvo['Pares Média'])
            dist_repetidas = abs(stats['repetidas'] - alvo['Repetidas Média'])
            score += (1 / (1 + dist_soma)) * pesos.get('peso_proximidade_soma', 0)
            score += (1 / (1 + dist_pares)) * pesos.get('peso_proximidade_pares', 0)
            score += (1 / (1 + dist_repetidas)) * pesos.get('peso_proximidade_repetidas', 0)
        score += stats['frias'] * pesos.get('peso_por_fria', 0)
        score += stats['quentes'] * pesos.get('peso_por_quente', 0)
        return score

    def _gerar_melhores_jogos(self, n_jogos, dados, estrategia):
        pesos = estrategia['pesos']
        alvo = estrategia.get('alvo')
        num_candidatos = self.config.get("NUM_CANDIDATOS", 10000)
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

    def _obter_analise_ia(self, jogos_com_score, dados, regras, perfil):
        if not self.model or not jogos_com_score: return None
        jogos_formatados = []
        for jogo, score in jogos_com_score:
            soma = sum(jogo); repetidas = len(set(jogo) & set(dados['concurso_anterior']))
            pares = len([n for n in jogo if n % 2 == 0]); frias = len(set(jogo) & set(dados['frias']))
            quentes = len(set(jogo) & set(dados['quentes']))
            jogos_formatados.append(f"- Jogo: {jogo}\n  - Pontuação: {score:.2f}\n  - Estatísticas: Soma={soma}, Repetidas={repetidas}, Pares={pares}, Frias={frias}, Quentes={quentes}")
        texto_jogos = "\n".join(jogos_formatados)
        prompt = f"""
        Aja como um especialista em análise de loterias. Sua tarefa é fornecer uma análise qualitativa e sucinta sobre jogos gerados por um algoritmo.
        **ESTRATÉGIA APLICADA:** {perfil}
        **METAS DO ALGORITMO:**
        - Dezenas Frias (observar): {dados['frias']}
        - Tendência Repetidas (meta): {regras['repetidas_ideal']} | Tendência Soma (meta): {regras['soma_ideal']}
        **JOGOS GERADOS E SUAS ESTATÍSTICAS:**
        {texto_jogos}
        **SUA ANÁLISE:**
        Explique por que estes jogos receberam uma alta pontuação, comparando suas estatísticas com as metas.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e: return f"Ocorreu um erro ao chamar a API: {e}"

    def gerar_jogos_com_analise(self, estrategia, n_jogos):
        st.toast("Analisando tendências do último concurso...")
        regras_dinamicas = self._analisar_tendencias(self.df)
        dados_futuro = {}
        dados_futuro['concurso_anterior'] = sorted([int(self.df.iloc[-1][f'Bola{i}']) for i in range(1, 16)])
        frequencias = pd.Series(self.df[[f'Bola{i}' for i in range(1, 16)]].values.flatten()).value_counts()
        dados_futuro['quentes'] = sorted(list(frequencias.head(8).index.astype(int)))
        dados_futuro['frias'] = self._get_dezenas_frias(self.df, n=8)
        st.toast(f"Gerando e pontuando jogos com a estratégia '{estrategia['nome']}'...")
        jogos_com_score = self._gerar_melhores_jogos(n_jogos, dados_futuro, estrategia)
        st.toast("Solicitando análise da IA do Gemini...")
        analise_ia = self._obter_analise_ia(jogos_com_score, dados_futuro, regras_dinamicas, estrategia['nome'])
        return {"jogos": jogos_com_score, "analise": analise_ia}

    def rodar_simulacao(self):
        concursos_para_testar = self.df.tail(self.config['NUMERO_DE_CONCURSOS_A_TESTAR'])
        resultados_por_estrategia = {est['nome']: {'resumo_acertos': {i: 0 for i in range(11, 16)}, 'custo': 0.0, 'retorno': 0.0} for est in self.estrategias}
        historico_lucro_acumulado = []; lucro_total_acumulado = 0.0
        historico_jogos_detalhado = []
        barra_progresso = st.progress(0, text="Simulando concursos...")
        for i, (_, concurso_alvo) in enumerate(concursos_para_testar.iterrows()):
            num_concurso_alvo = int(concurso_alvo['Concurso'])
            df_historico = self.df[self.df['Concurso'] < num_concurso_alvo]
            if df_historico.empty: continue
            resultado_real = sorted([int(concurso_alvo[f'Bola{i}']) for i in range(1, 16)])
            dados = {'concurso_anterior': sorted([int(df_historico.iloc[-1][f'Bola{i}']) for i in range(1, 16)]), 'quentes': [], 'frias': self._get_dezenas_frias(df_historico, n=8)}
            custo_neste_concurso = 0.0; retorno_neste_concurso = 0.0
            jogos_deste_concurso_para_df = []
            for estrategia in self.estrategias:
                jogos_por_estrategia = self.config['JOGOS_POR_SIMULACAO'] // len(self.estrategias)
                if jogos_por_estrategia == 0: jogos_por_estrategia = 1
                jogos_gerados_com_score = self._gerar_melhores_jogos(jogos_por_estrategia, dados, estrategia)
                for jogo, score in jogos_gerados_com_score:
                    acertos = len(set(jogo) & set(resultado_real))
                    jogos_deste_concurso_para_df.append({"estrategia": estrategia['nome'], "jogo": ", ".join(map(str, jogo)), "acertos": acertos})
                    if acertos >= 11 and acertos <= 15:
                        resultados_por_estrategia[estrategia['nome']]['resumo_acertos'][acertos] += 1
                    premio = PREMIOS_FIXOS.get(acertos, 0.0)
                    resultados_por_estrategia[estrategia['nome']]['custo'] += CUSTO_JOGO_15_DEZENAS
                    custo_neste_concurso += CUSTO_JOGO_15_DEZENAS
                    if premio > 0:
                        resultados_por_estrategia[estrategia['nome']]['retorno'] += premio
                        retorno_neste_concurso += premio
            historico_jogos_detalhado.append({"concurso_n": num_concurso_alvo, "resultado_real": ", ".join(map(str, resultado_real)), "jogos_gerados": pd.DataFrame(jogos_deste_concurso_para_df)})
            lucro_neste_concurso = retorno_neste_concurso - custo_neste_concurso
            lucro_total_acumulado += lucro_neste_concurso
            historico_lucro_acumulado.append(lucro_total_acumulado)
            barra_progresso.progress((i + 1) / len(concursos_para_testar), text=f"Simulando Concurso {num_concurso_alvo}")
        barra_progresso.empty()
        custo_total = sum(v['custo'] for v in resultados_por_estrategia.values())
        retorno_total = sum(v['retorno'] for v in resultados_por_estrategia.values())
        lucro_total = retorno_total - custo_total
        roi_total = (lucro_total / custo_total * 100) if custo_total > 0 else 0
        return {
            "resultados_por_estrategia": resultados_por_estrategia,
            "historico_lucro": historico_lucro_acumulado,
            "historico_detalhado": historico_jogos_detalhado,
            "metricas_gerais": {"custo_total": custo_total, "retorno_total": retorno_total, "lucro_total": lucro_total, "roi_total": roi_total}
        }

    def descobrir_perfis(self, n_perfis):
        df_analise = self.df.copy()
        bola_cols = [f'Bola{i}' for i in range(1, 16)]
        df_anterior = self.df[bola_cols].shift(1)
        def calcular_repetidas(row):
            if pd.isna(row.iloc[-15:].sum()): return 0
            concurso_atual = set(row.iloc[:15].values)
            concurso_anterior = set(row.iloc[-15:].values)
            return len(concurso_atual & concurso_anterior)
        df_merged = pd.concat([df_analise[bola_cols], df_anterior.rename(columns=lambda c: f"{c}_ant")], axis=1)
        df_analise['repetidas'] = df_merged.apply(calcular_repetidas, axis=1)
        df_analise['soma'] = df_analise[bola_cols].sum(axis=1)
        df_analise['pares'] = df_analise[bola_cols].apply(lambda row: sum(1 for n in row if n % 2 == 0), axis=1)
        features_df = df_analise[['soma', 'pares', 'repetidas']].iloc[1:]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        kmeans = KMeans(n_clusters=n_perfis, random_state=42, n_init='auto')
        features_df['perfil_descoberto'] = kmeans.fit_predict(features_scaled)
        centros_reais = scaler.inverse_transform(kmeans.cluster_centers_)
        perfis_analisados = []
        for i, centro in enumerate(centros_reais):
            perfil = {"Soma Média": round(centro[0]),"Pares Média": round(centro[1]),"Repetidas Média": round(centro[2]),"Ocorrências": int(features_df['perfil_descoberto'].value_counts()[i])}
            perfis_analisados.append(perfil)
        return perfis_analisados, features_df['perfil_descoberto'].value_counts()
    
    def _backtest_silencioso(self, pesos, alvo):
        concursos_para_testar = self.df.tail(self.config['NUMERO_DE_CONCURSOS_A_TESTAR'])
        custo_total = 0.0; retorno_total = 0.0
        estrategia_teste = {"pesos": pesos, "alvo": alvo}
        for _, concurso_alvo in concursos_para_testar.iterrows():
            df_historico = self.df[self.df['Concurso'] < int(concurso_alvo['Concurso'])]
            if df_historico.empty: continue
            resultado_real = sorted([int(concurso_alvo[f'Bola{i}']) for i in range(1, 16)])
            dados = {'concurso_anterior': sorted([int(df_historico.iloc[-1][f'Bola{i}']) for i in range(1, 16)]), 'quentes': [], 'frias': self._get_dezenas_frias(df_historico, n=8)}
            jogos_gerados_com_score = self._gerar_melhores_jogos(self.config['JOGOS_POR_SIMULACAO'], dados, estrategia_teste)
            for jogo, score in jogos_gerados_com_score:
                custo_total += CUSTO_JOGO_15_DEZENAS
                acertos = len(set(jogo) & set(resultado_real))
                retorno_total += PREMIOS_FIXOS.get(acertos, 0.0)
        roi = ((retorno_total - custo_total) / custo_total * 100) if custo_total > 0 else -100
        return roi

    def otimizar_estrategia(self, perfil_alvo, status_placeholder):
        search_space = self.search_spaces.get("Descoberto")
        @use_named_args(search_space)
        def funcao_objetivo(**params):
            roi = self._backtest_silencioso(params, perfil_alvo)
            iteracao_atual = len(resultados_parciais) + 1
            log_message = f"Teste {iteracao_atual}/{self.config['OTIMIZACAO_CHAMADAS']} -> ROI: {roi:.2f}%"
            print(log_message)
            with status_placeholder:
                st.write(log_message)
            resultados_parciais.append(roi)
            return -roi
        
        resultados_parciais = []
        resultado_otimizacao = gp_minimize(func=funcao_objetivo, dimensions=search_space, n_calls=self.config['OTIMIZACAO_CHAMADAS'], random_state=42)
        melhor_roi = -resultado_otimizacao.fun
        melhores_pesos = {dim.name: int(val) for dim, val in zip(search_space, resultado_otimizacao.x)}
        return {"roi": melhor_roi, "pesos": melhores_pesos}

# ==============================================================================
# --- FUNÇÕES DE APOIO E INTERFACE ---
# ==============================================================================
@st.cache_data(ttl=3600)
# app.py -> Substitua a função carregar_dados por esta:

@st.cache_data(ttl=3600)
def carregar_dados():
    # Carrega as credenciais a partir dos Secrets do Streamlit (o método correto para a nuvem)
    creds_dict = json.loads(st.secrets["GSPREAD_CREDENTIALS"])
    gc = gspread.service_account_from_dict(creds_dict)
    
    spreadsheet = gc.open_by_url("https://docs.google.com/spreadsheets/d/1cc-JxB_-FkOEIeD_t2SZ9RFpKr3deUfCFuTR_2wXCYk/edit?gid=498647709#gid=498647709")
    worksheet = spreadsheet.sheet1
    rows = worksheet.get_all_records()
    df = pd.DataFrame(rows)
    
    # Processamento do DataFrame
    bola_cols = [f'Bola{i}' for i in range(1, 16)]
    for col in ['Concurso'] + bola_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Concurso'] + bola_cols, inplace=True)
    df[bola_cols] = df[bola_cols].astype(int)
    df['Concurso'] = df['Concurso'].astype(int)
    df.sort_values(by='Concurso', inplace=True)
    return df
def carregar_modelo_ia():
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        return model
    except Exception as e:
        st.error(f"Erro ao configurar o modelo de IA. Verifique seu 'secrets.toml'. Detalhes: {e}")
        return None
def gerar_graficos(resultados_por_estrategia, historico_lucro):
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    nomes_estrategias = list(resultados_por_estrategia.keys())
    lucros = [v['retorno'] - v['custo'] for v in resultados_por_estrategia.values()]
    cores = ['royalblue' if l >= 0 else 'salmon' for l in lucros]
    bars = ax1.bar(nomes_estrategias, lucros, color=cores)
    ax1.set_ylabel('Lucro / Prejuízo (R$)'); ax1.set_title('Resultado Financeiro Final por Estratégia')
    ax1.axhline(0, color='black', linewidth=0.8); ax1.bar_label(bars, fmt='R$ %.2f')
    plt.xticks(rotation=45, ha="right")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(historico_lucro, marker='o', linestyle='-', color='green')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.2, label='Ponto de Equilíbrio (R$ 0)')
    ax2.set_title('Evolução do Saldo Financeiro Acumulado'); ax2.set_xlabel('Concursos Simulados')
    ax2.set_ylabel('Saldo Acumulado (R$)'); ax2.grid(True, which='both', linestyle='--', linewidth=0.5); ax2.legend()
    return fig1, fig2

st.set_page_config(layout="wide", page_title="Analisador Lotofácil com IA")
st.title("🤖 Analisador Inteligente da Lotofácil")

st.sidebar.title("Ferramentas")
modo_app = st.sidebar.radio(
    "Escolha o que deseja fazer:",
    ["Gerador de Jogos", "Simulador de Estratégias", "Laboratório de IA"],
    key="navigation"
)

try:
    df = carregar_dados()
    model = carregar_modelo_ia() if modo_app == "Gerador de Jogos" else None
    st.sidebar.success(f"Base de dados carregada! \n\nÚltimo concurso: {df['Concurso'].max()}")
    with st.sidebar.expander("Ver Estratégias Salvas"):
        estrategias_atuais = carregar_estrategias()
        for i, est in enumerate(estrategias_atuais):
            st.write(f"**{i+1}. {est['nome']}** (ROI: {est['roi']:.2f}%)")
except Exception as e:
    st.error(f"Ocorreu um erro fatal no carregamento: {e}"); st.stop()

if modo_app == "Gerador de Jogos":
    st.header("Gerador de Jogos para o Próximo Concurso")
    st.sidebar.header("Configurações de Geração")
    estrategias_salvas = carregar_estrategias()
    nomes_estrategias = [est['nome'] for est in estrategias_salvas]
    estrategia_selecionada_nome = st.sidebar.selectbox("Escolha a Estratégia a ser usada:", nomes_estrategias)
    jogos_a_gerar = st.sidebar.slider("Quantos jogos você deseja gerar?", min_value=1, max_value=30, value=6, step=1)
    if st.sidebar.button("Gerar Jogos Inteligentes", type="primary", use_container_width=True):
        estrategia_obj = next((item for item in estrategias_salvas if item["nome"] == estrategia_selecionada_nome), None)
        if estrategia_obj:
            CONFIG = {"NUM_CANDIDATOS": 20000}
            analisador = LotofacilAnalisador(df, CONFIG, model)
            with st.spinner(f"Gerando jogos com a estratégia '{estrategia_obj['nome']}'..."):
                resultados = analisador.gerar_jogos_com_analise(estrategia_obj, jogos_a_gerar)
            st.subheader(f"Resultados da Estratégia: {estrategia_obj['nome']}")
            df_jogos = pd.DataFrame([jogo for jogo, score in resultados["jogos"]], columns=[f'D{i+1}' for i in range(15)])
            st.dataframe(df_jogos.style.set_properties(**{'text-align': 'center', 'font-size': '13px'}).hide(axis="index"), use_container_width=True)
            with st.expander("Ver Análise da IA para esta Estratégia"):
                if resultados["analise"]: st.markdown(resultados["analise"])
        else:
            st.error("Estratégia selecionada não foi encontrada.")
    else:
        st.info("Ajuste as configurações e clique no botão para gerar os jogos.")

elif modo_app == "Simulador de Estratégias":
    st.header("Simulador de Estratégias (Backtest Financeiro)")
    st.sidebar.header("Configurações da Simulação")
    concursos_a_testar = st.sidebar.slider("Quantos concursos testar?", min_value=10, max_value=500, value=50, step=10)
    jogos_por_simulacao = st.sidebar.slider("Quantos jogos simular por concurso?", min_value=1, max_value=50, value=10, step=1)
    if st.sidebar.button("Rodar Simulação", type="primary", use_container_width=True):
        CONFIG = {"NUMERO_DE_CONCURSOS_A_TESTAR": concursos_a_testar, "JOGOS_POR_SIMULACAO": jogos_por_simulacao, "DEZENAS_POR_JOGO": 15, "NUM_CANDIDATOS": 10000, "ESTRATEGIAS_ATUAIS": carregar_estrategias()}
        analisador = LotofacilAnalisador(df, CONFIG)
        resultados = analisador.rodar_simulacao()
        st.subheader("Resultados Financeiros da Simulação")
        m = resultados['metricas_gerais']
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Lucro / Prejuízo Total", f"R$ {m['lucro_total']:.2f}", f"{m['roi_total']:.2f}% ROI")
        kpi2.metric("Custo Total", f"R$ {m['custo_total']:.2f}")
        kpi3.metric("Retorno Total", f"R$ {m['retorno_total']:.2f}")
        st.markdown("---")
        st.subheader("Análise Detalhada por Estratégia")
        res_est = resultados['resultados_por_estrategia']
        cols = st.columns(len(res_est) if res_est else 1)
        for i, (nome_est, dados_est) in enumerate(res_est.items()):
            with cols[i]:
                st.markdown(f"##### {nome_est}")
                lucro = dados_est['retorno'] - dados_est['custo']
                roi = (lucro / dados_est['custo'] * 100) if dados_est['custo'] > 0 else 0
                st.metric("Lucro / Prejuízo", f"R$ {lucro:.2f}", f"{roi:.2f}% ROI")
                st.markdown("**Resumo de Acertos:**")
                df_acertos = pd.DataFrame(list(dados_est['resumo_acertos'].items()), columns=['Acertos', 'Quantidade']).sort_values(by='Acertos')
                df_acertos = df_acertos[df_acertos['Quantidade'] > 0].set_index('Acertos')
                st.dataframe(df_acertos.style.set_properties(**{'text-align': 'center', 'font-size': '13px'}), use_container_width=True)
        st.subheader("Gráficos de Desempenho")
        fig1, fig2 = gerar_graficos(res_est, resultados['historico_lucro'])
        st.pyplot(fig1)
        st.pyplot(fig2)
        with st.expander("Ver Detalhes dos Jogos Simulados (do mais recente ao mais antigo)"):
            for resultado_concurso in reversed(resultados['historico_detalhado']):
                st.markdown(f"#### Concurso: **{resultado_concurso['concurso_n']}**")
                st.markdown(f"**Resultado Real:** {resultado_concurso['resultado_real']}")
                df_jogos = resultado_concurso['jogos_gerados']
                def colorir_premios(val):
                    color = '#2E8B57' if val >= 11 else ''
                    return f'background-color: {color}'
                st.dataframe(df_jogos.style.apply(lambda x: x.map(colorir_premios), subset=['acertos']).set_properties(**{'text-align': 'center', 'font-size': '13px'}).hide(axis="index"), use_container_width=True)
    else:
        st.info("Ajuste as configurações da simulação na barra lateral e clique em 'Rodar Simulação' para começar.")

elif modo_app == "Laboratório de IA":
    st.header("🔬 Laboratório de IA: Descoberta e Otimização")
    st.info("Esta ferramenta automatiza a descoberta de novos perfis de jogo e a otimização de estratégias para eles.")
    st.warning("**Atenção:** Este processo é extremamente demorado e pode levar horas.", icon="⏳")
    st.sidebar.header("Configurações do Laboratório")
    n_perfis = st.sidebar.number_input("Quantos perfis a IA deve procurar?", min_value=2, max_value=10, value=3, step=1)
    chamadas_otimizacao = st.sidebar.slider("Qualidade da Otimização (por perfil)", min_value=10, max_value=200, value=20, step=5)
    if st.sidebar.button("Iniciar Descoberta e Otimização", type="primary", use_container_width=True):
        CONFIG = {"NUMERO_DE_CONCURSOS_A_TESTAR": 100, "JOGOS_POR_SIMULACAO": 10, "NUM_CANDIDATOS": 5000, "OTIMIZACAO_CHAMADAS": chamadas_otimizacao}
        analisador = LotofacilAnalisador(df, CONFIG)
        with st.status("Iniciando processo completo...", expanded=True) as status:
            st.write("Fase 1 de 2: Descobrindo perfis de jogo no histórico...")
            perfis_descobertos, _ = analisador.descobrir_perfis(n_perfis)
            st.success(f"{len(perfis_descobertos)} perfis de jogo foram descobertos!")
            st.json(perfis_descobertos)
            st.write("\nFase 2 de 2: Otimizando uma estratégia para cada perfil descoberto...")
            estrategias_atuais = carregar_estrategias()
            for i, perfil_alvo in enumerate(perfis_descobertos):
                nome_perfil = f"Descoberto {i+1}: Soma {perfil_alvo['Soma Média']}, Pares {perfil_alvo['Pares Média']}, Rep {perfil_alvo['Repetidas Média']}"
                status.update(label=f"Otimizando perfil {i+1}/{len(perfis_descobertos)}: {nome_perfil}")
                resultado_opt = analisador.otimizar_estrategia(perfil_alvo, status)
                novo_roi = resultado_opt['roi']
                novos_pesos = resultado_opt['pesos']
                st.write(f"Otimização para '{nome_perfil}' concluída com ROI de {novo_roi:.2f}%. Validando...")
                rois_salvos = [est['roi'] for est in estrategias_atuais]
                pior_roi_salvo = min(rois_salvos) if rois_salvos else -999
                if novo_roi > pior_roi_salvo:
                    nova_estrategia = {"nome": nome_perfil, "tipo": "Descoberto", "roi": novo_roi, "alvo": perfil_alvo, "pesos": novos_pesos}
                    pior_indice = np.argmin(rois_salvos) if rois_salvos else -1
                    if len(estrategias_atuais) >= 4 and pior_indice != -1 :
                        st.write(f"-> Nova estratégia superou '{estrategias_atuais[pior_indice]['nome']}' (ROI {pior_roi_salvo:.2f}%). Substituindo...")
                        estrategias_atuais[pior_indice] = nova_estrategia
                    else:
                        estrategias_atuais.append(nova_estrategia)
                    salvar_estrategias(estrategias_atuais)
                    st.success(f"Estratégia '{nome_perfil}' (ROI {novo_roi:.2f}%) foi salva!")
                else:
                    st.warning(f"Estratégia para '{nome_perfil}' (ROI {novo_roi:.2f}%) não superou as existentes e foi descartada.")
            status.update(label="Processo concluído!", state="complete")
        st.balloons()
        st.header("Processo Finalizado!")
        st.info("As melhores estratégias descobertas foram salvas. Elas já estão disponíveis no 'Gerador de Jogos'. Atualize a página (F5) para usar.")
    else:
        st.info("Ajuste as configurações e clique no botão para iniciar o processo de descoberta e otimização.")
