# app_improved.py - Versão otimizada do Analisador Lotofácil (v1.0)
# Melhorias:
# - Vetorização de cálculo de dezenas frias/quentes
# - heapq para manter apenas os melhores candidatos (memória)
# - Função centralizada de avaliação de jogos
# - Tratamento de erros consistente
# - Constantes centralizadas
# - Exportação de resultados CSV/JSON
# - Prompt IA com fallback local

import os
import json
import math
import heapq
import random
import re
import statistics
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# libs já usadas no original — mantenho por compatibilidade
try:
    import gspread
except Exception:
    gspread = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# -----------------------------
# CONSTANTES / CONFIGURAÇÃO
# -----------------------------
ESTRATEGIAS_FILE = "estrategias.json"
CUSTO_JOGO_15_DEZENAS = 3.50
PREMIOS_FIXOS = {11: 7.0, 12: 14.0, 13: 35.0}  # prêmios fixos que você usava
DEZENAS_TOTAL = 25
DEFAULT_NUM_FRIAS = 8
DEFAULT_NUM_QUENTES = 8
DEFAULT_NUM_CANDIDATOS = 10000
PROGRESS_UPDATE_BATCH = 50  # atualiza barra de progresso a cada N iterações
CACHE_TTL = 3600  # segundos

# conjuntos fixos (mantive os originais)
PRIMOS = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
UNIVERSO = list(range(1, DEZENAS_TOTAL + 1))

# -----------------------------
# UTILITÁRIOS / I/O
# -----------------------------
def carregar_estrategias() -> List[Dict]:
    if os.path.exists(ESTRATEGIAS_FILE):
        try:
            with open(ESTRATEGIAS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # defaults se arquivo inexistente/corrompido
    return [
        {"nome": "Padrão (Manual)", "tipo": "Manual", "roi": -70.0,
         "alvo": {'Soma Média': 195, 'Pares Média': 8, 'Repetidas Média': 9, 'Primos Média': 5, 'Moldura Média': 10},
         "pesos": {'peso_proximidade_soma': 20, 'peso_proximidade_pares': 10, 'peso_proximidade_repetidas': 20,
                   'peso_por_fria': 15, 'peso_por_quente': 5, 'peso_proximidade_primos': 5, 'peso_proximidade_moldura': 5}},
        {"nome": "Contrário (Manual)", "tipo": "Manual", "roi": -70.0,
         "alvo": {'Soma Média': 180, 'Pares Média': 7, 'Repetidas Média': 7, 'Primos Média': 6, 'Moldura Média': 11},
         "pesos": {'peso_proximidade_soma': 10, 'peso_proximidade_pares': 5, 'peso_proximidade_repetidas': 10,
                   'peso_por_fria': 25, 'peso_por_quente': -10, 'peso_proximidade_primos': 0, 'peso_proximidade_moldura': 0}}
    ]

def salvar_estrategias(lista):
    with open(ESTRATEGIAS_FILE, "w", encoding="utf-8") as f:
        json.dump(lista, f, indent=4, ensure_ascii=False)

@st.cache_data(ttl=CACHE_TTL)
def carregar_dados_google_sheets() -> pd.DataFrame:
    """
    Carrega resultados da planilha do Google via gspread.
    Mantive a implementação original, mas com tratamento de erro claro.
    """
    if gspread is None:
        raise RuntimeError("Biblioteca 'gspread' não está disponível no ambiente.")
    # espera que st.secrets tenha as credenciais
    creds_dict = json.loads(st.secrets["GSPREAD_CREDENTIALS"])
    gc = gspread.service_account_from_dict(creds_dict)
    spreadsheet = gc.open_by_url(st.secrets.get("GSPREAD_SHEET_URL"))
    worksheet = spreadsheet.sheet1
    rows = worksheet.get_all_records()
    df = pd.DataFrame(rows)
    bola_cols = [f'Bola{i}' for i in range(1, 16)]
    for col in ['Concurso'] + bola_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Concurso'] + bola_cols, inplace=True)
    df[bola_cols] = df[bola_cols].astype(int)
    df['Concurso'] = df['Concurso'].astype(int)
    df.sort_values(by='Concurso', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_resource
def configurar_modelo_ia():
    """
    Tenta configurar o modelo Gemini (se disponível). Retorna None se não for possível.
    """
    if genai is None:
        return None
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        return model
    except Exception:
        return None

# -----------------------------
# FUNÇÕES DE CÁLCULO OTIMIZADAS
# -----------------------------
def calcular_ultimos_aparecimentos(df: pd.DataFrame) -> Dict[int, int]:
    """
    Retorna último índice (Concurso) em que cada dezena apareceu.
    Implementação vetorizada: melt e groupby para extrair última ocorrência.
    """
    bola_cols = [f'Bola{i}' for i in range(1, 16)]
    # cria DF longo: coluna 'dezena' com o valor e 'Concurso'
    df_long = df[['Concurso'] + bola_cols].melt(id_vars='Concurso', value_vars=bola_cols, value_name='dezena')
    # último concurso onde cada dezena apareceu
    last = df_long.groupby('dezena', as_index=False)['Concurso'].max()
    ultimo_por_dezena = {int(row['dezena']): int(row['Concurso']) for _, row in last.iterrows()}
    # para dezenas que nunca apareceram em dataset, consideramos 0
    for d in UNIVERSO:
        if d not in ultimo_por_dezena:
            ultimo_por_dezena[d] = 0
    return ultimo_por_dezena

def obter_dezenas_frias(df: pd.DataFrame, n: int = DEFAULT_NUM_FRIAS) -> List[int]:
    """
    Ordena por maior atraso (diferença entre último concurso conhecido e último aparecimento).
    Vetorizado usando calcular_ultimos_aparecimentos.
    """
    if df.empty:
        return sorted(UNIVERSO)[:n]
    ultimo_concurso_geral = int(df['Concurso'].max())
    ultimo_por_dezena = calcular_ultimos_aparecimentos(df)
    atrasos = {d: ultimo_concurso_geral - ultimo_por_dezena.get(d, 0) for d in UNIVERSO}
    # sort por atraso desc (maior atraso = mais "fria")
    arr = sorted(atrasos.items(), key=lambda x: x[1], reverse=True)
    return [int(x[0]) for x in arr[:n]]

def obter_dezenas_quentes(df: pd.DataFrame, n: int = DEFAULT_NUM_QUENTES) -> List[int]:
    """
    Frequências das dezenas nas últimas N concursos (ou no histórico todo).
    """
    if df.empty:
        return sorted(UNIVERSO)[:n]
    bola_cols = [f'Bola{i}' for i in range(1, 16)]
    freqs = pd.Series(df[bola_cols].values.flatten()).value_counts()
    top = freqs.head(n).index.astype(int).tolist()
    # caso faltando (por segurança), complete com as menores dezenas
    if len(top) < n:
        for d in UNIVERSO:
            if d not in top:
                top.append(d)
            if len(top) == n:
                break
    return top

# -----------------------------
# AVALIAÇÃO DE JOGOS (centralizada)
# -----------------------------
def avaliar_jogo(jogo: List[int], dados: Dict, pesos: Dict, alvo: Optional[Dict]) -> Tuple[float, Dict]:
    """
    Calcula score e estatísticas do jogo com base em pesos e metas.
    Retorna (score, stats)
    """
    soma = sum(jogo)
    pares = sum(1 for n in jogo if n % 2 == 0)
    repetidas = len(set(jogo) & set(dados.get('concurso_anterior', [])))
    frias = len(set(jogo) & set(dados.get('frias', [])))
    quentes = len(set(jogo) & set(dados.get('quentes', [])))
    primos = len(set(jogo) & PRIMOS)
    moldura = len(set(jogo) & MOLDURA)
    stats = {'soma': soma, 'pares': pares, 'repetidas': repetidas,
             'frias': frias, 'quentes': quentes, 'primos': primos, 'moldura': moldura}
    score = 0.0
    if alvo:
        dist_soma = abs(soma - alvo.get('Soma Média', 195))
        dist_pares = abs(pares - alvo.get('Pares Média', 8))
        dist_repetidas = abs(repetidas - alvo.get('Repetidas Média', 9))
        dist_primos = abs(primos - alvo.get('Primos Média', 5))
        dist_moldura = abs(moldura - alvo.get('Moldura Média', 10))
        score += (1.0 / (1 + dist_soma)) * pesos.get('peso_proximidade_soma', 0)
        score += (1.0 / (1 + dist_pares)) * pesos.get('peso_proximidade_pares', 0)
        score += (1.0 / (1 + dist_repetidas)) * pesos.get('peso_proximidade_repetidas', 0)
        score += (1.0 / (1 + dist_primos)) * pesos.get('peso_proximidade_primos', 0)
        score += (1.0 / (1 + dist_moldura)) * pesos.get('peso_proximidade_moldura', 0)
    score += frias * pesos.get('peso_por_fria', 0)
    score += quentes * pesos.get('peso_por_quente', 0)
    return score, stats

# -----------------------------
# GERAÇÃO DE CANDIDATOS EFICIENTE (heap)
# -----------------------------
def gerar_melhores_jogos(n_jogos: int, dados: Dict, estrategia: Dict, num_candidatos: int = DEFAULT_NUM_CANDIDATOS) -> List[Tuple[List[int], float]]:
    """
    Gera 'num_candidatos' candidatos aleatórios e mantém apenas os top n_jogos usando heapq.
    Isso reduz memória e custo de sort completo.
    """
    pesos = estrategia.get('pesos', {})
    alvo = estrategia.get('alvo', None)
    # min-heap com tuplas (score, jogo); usaremos score negativo para ordenar maior primeiro via heapq
    heap = []
    seen = set()
    batch = 0
    for i in range(num_candidatos):
        jogo = tuple(sorted(random.sample(UNIVERSO, 15)))
        if jogo in seen:
            continue
        seen.add(jogo)
        score, _ = avaliar_jogo(list(jogo), dados, pesos, alvo)
        if len(heap) < n_jogos:
            heapq.heappush(heap, (score, list(jogo)))
        else:
            # se score maior que menor do heap, substitui
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, list(jogo)))
        batch += 1
    # retornar em ordem decrescente
    melhores = sorted(heap, key=lambda x: x[0], reverse=True)
    return [(jogo, float(score)) for score, jogo in melhores]

# -----------------------------
# CLASSE PRINCIPAL (refatorada)
# -----------------------------
class LotofacilAnalisador:
    def __init__(self, df: pd.DataFrame, config: Dict = None, model=None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.config = config or {}
        self.model = model
        self.estrategias = self.config.get("ESTRATEGIAS_ATUAIS", carregar_estrategias())
        # search_spaces (mantido para otimização)
        self.search_spaces = {
            "Descoberto": [
                Integer(0, 100, name='peso_proximidade_soma'), Integer(0, 100, name='peso_proximidade_pares'),
                Integer(0, 100, name='peso_proximidade_repetidas'), Integer(0, 50, name='peso_proximidade_primos'),
                Integer(0, 50, name='peso_proximidade_moldura'), Integer(0, 50, name='peso_por_fria'),
                Integer(-50, 0, name='peso_por_quente')
            ]
        }

    def _analisar_tendencias(self, n_concursos: int = 15) -> Dict:
        """
        Retorna regras dinâmicas (soma média e repetidas ideal) usando os últimos n_concursos.
        """
        if self.df.empty:
            return {'repetidas_ideal': 9, 'soma_ideal': 195}
        df_t = self.df.tail(n_concursos)
        bola_cols = [f'Bola{i}' for i in range(1, 16)]
        repetidas = []
        # cálculo de repetidas entre concursos consecutivos
        arr = df_t[bola_cols].values
        for i in range(len(arr)-1):
            repetidas.append(len(set(arr[i]) & set(arr[i+1])))
        regra_repetidas = int(statistics.mode(repetidas)) if repetidas else 9
        somas = df_t[bola_cols].sum(axis=1).tolist()
        soma_media = int(statistics.mean(somas)) if somas else 195
        return {'repetidas_ideal': regra_repetidas, 'soma_ideal': soma_media}

    def gerar_jogos_com_analise(self, estrategia: Dict, n_jogos: int):
        """
        Gera jogos, chama análise de IA (se disponível) e retorna jogos + análise.
        """
        regras = self._analisar_tendencias()
        dados_futuro = {}
        if not self.df.empty:
            ultimo = self.df.iloc[-1]
            dados_futuro['concurso_anterior'] = sorted([int(ultimo[f'Bola{i}']) for i in range(1, 16)])
            dados_futuro['quentes'] = obter_dezenas_quentes(self.df, DEFAULT_NUM_QUENTES)
            dados_futuro['frias'] = obter_dezenas_frias(self.df, DEFAULT_NUM_FRIAS)
        else:
            dados_futuro['concurso_anterior'] = []
            dados_futuro['quentes'] = []
            dados_futuro['frias'] = []
        jogos_com_score = gerar_melhores_jogos(n_jogos, dados_futuro, estrategia,
                                              num_candidatos=self.config.get("NUM_CANDIDATOS", DEFAULT_NUM_CANDIDATOS))
        analise_ia = self._obter_analise_ia(jogos_com_score, dados_futuro, regras, estrategia.get('nome', 'Estratégia'))
        return {"jogos": jogos_com_score, "analise": analise_ia}

    def _obter_analise_ia(self, jogos_com_score, dados, regras, perfil):
        """
        Tenta usar o modelo de IA; se não disponível, retorna um resumo local gerado programaticamente.
        """
        if not self.model or not jogos_com_score:
            return self._analise_local(jogos_com_score, dados, regras, perfil)
        # cria prompt sucinto e seguro
        jogos_texto = []
        for jogo, score in jogos_com_score:
            soma = sum(jogo); pares = sum(1 for n in jogo if n % 2 == 0)
            repetidas = len(set(jogo) & set(dados.get('concurso_anterior', [])))
            jogos_texto.append(f"Jogo: {jogo} | Score: {score:.2f} | Soma: {soma} | Pares: {pares} | Repetidas: {repetidas}")
        prompt = (
            f"Aja como um especialista conciso em análise de jogos de loteria.\n"
            f"Estratégia: {perfil}\n"
            f"Metas: repetidas={regras['repetidas_ideal']}, soma={regras['soma_ideal']}\n"
            f"Dados frias: {dados.get('frias', [])} | quentes: {dados.get('quentes',[])}\n"
            f"Jogos:\n" + "\n".join(jogos_texto) + "\n\nForneça 4-6 linhas explicando por que cada jogo pontuou bem e riscos."
        )
        try:
            resp = self.model.generate_content(prompt)
            return resp.text
        except Exception:
            # fallback local
            return self._analise_local(jogos_com_score, dados, regras, perfil)

    def _analise_local(self, jogos_com_score, dados, regras, perfil):
        """
        Gera uma análise local simples caso IA indisponível.
        """
        lines = [f"Análise automática (fallback) para a estratégia '{perfil}':"]
        for jogo, score in jogos_com_score[:6]:
            soma = sum(jogo); pares = sum(1 for n in jogo if n % 2 == 0)
            repetidas = len(set(jogo) & set(dados.get('concurso_anterior', [])))
            lines.append(f"- Jogo {jogo}: score {score:.2f} | soma {soma} | pares {pares} | repetidas {repetidas}")
        lines.append("Observação: análise baseada em métricas locais; para explicações qualitativas mais profundas, conecte um modelo de IA.")
        return "\n".join(lines)

    def rodar_simulacao(self):
        """
        Backtest simplificado e mais eficiente.
        """
        num_concursos = int(self.config.get('NUMERO_DE_CONCURSOS_A_TESTAR', 50))
        concursos_para_testar = self.df.tail(num_concursos)
        resultados_por_estrategia = {est['nome']: {'resumo_acertos': {i: 0 for i in range(11, 16)}, 'custo': 0.0, 'retorno': 0.0} for est in self.estrategias}
        historico_lucro = []
        lucro_total = 0.0

        total_iters = len(concursos_para_testar)
        prog = st.progress(0, text="Simulando concursos...") if total_iters else None
        for i, (_, concurso) in enumerate(concursos_para_testar.iterrows()):
            num_concurso = int(concurso['Concurso'])
            df_hist = self.df[self.df['Concurso'] < num_concurso]
            if df_hist.empty:
                if prog: prog.progress((i + 1) / total_iters)
                continue
            resultado_real = sorted([int(concurso[f'Bola{j}']) for j in range(1, 16)])
            dados = {
                'concurso_anterior': sorted([int(df_hist.iloc[-1][f'Bola{j}']) for j in range(1, 16)]),
                'quentes': obter_dezenas_quentes(df_hist, DEFAULT_NUM_QUENTES),
                'frias': obter_dezenas_frias(df_hist, DEFAULT_NUM_FRIAS)
            }
            custo_concurso = 0.0
            retorno_concurso = 0.0
            for estrategia in self.estrategias:
                jogos_por_estrategia = max(1, int(self.config.get('JOGOS_POR_SIMULACAO', 10)) // max(1, len(self.estrategias)))
                jogos_gerados = gerar_melhores_jogos(jogos_por_estrategia, dados, estrategia,
                                                    num_candidatos=self.config.get('NUM_CANDIDATOS', 5000))
                for jogo, score in jogos_gerados:
                    acertos = len(set(jogo) & set(resultado_real))
                    if 11 <= acertos <= 15:
                        resultados_por_estrategia[estrategia['nome']]['resumo_acertos'][acertos] += 1
                    premio = PREMIOS_FIXOS.get(acertos, 0.0)
                    resultados_por_estrategia[estrategia['nome']]['custo'] += CUSTO_JOGO_15_DEZENAS
                    custo_concurso += CUSTO_JOGO_15_DEZENAS
                    if premio > 0:
                        resultados_por_estrategia[estrategia['nome']]['retorno'] += premio
                        retorno_concurso += premio
            lucro_n = retorno_concurso - custo_concurso
            lucro_total += lucro_n
            historico_lucro.append(lucro_total)
            if prog:
                prog.progress((i + 1) / total_iters, text=f"Simulando concurso {num_concurso}")
        if prog: prog.empty()
        custo_total = sum(v['custo'] for v in resultados_por_estrategia.values())
        retorno_total = sum(v['retorno'] for v in resultados_por_estrategia.values())
        lucro_total_final = retorno_total - custo_total
        roi_total = (lucro_total_final / custo_total * 100) if custo_total > 0 else 0.0
        return {"resultados_por_estrategia": resultados_por_estrategia, "historico_lucro": historico_lucro,
                "historico_detalhado": [], "metricas_gerais": {"custo_total": custo_total, "retorno_total": retorno_total,
                                                              "lucro_total": lucro_total_final, "roi_total": roi_total}}

    def descobrir_perfis(self, n_perfis: int = 3):
        """
        Descobre perfis com clusterização KMeans sobre features simples.
        """
        if self.df.empty:
            return [], pd.Series(dtype=int)
        bola_cols = [f'Bola{i}' for i in range(1, 16)]
        df2 = self.df.copy()
        df2['soma'] = df2[bola_cols].sum(axis=1)
        df2['pares'] = df2[bola_cols].apply(lambda row: sum(1 for n in row if n % 2 == 0), axis=1)
        df2['repetidas'] = df2[bola_cols].shift(1).combine(df2[bola_cols], lambda a, b: len(set(a) & set(b)) if a.notnull().all() else np.nan, axis=1)
        df2['primos'] = df2[bola_cols].apply(lambda row: len(set(row) & PRIMOS), axis=1)
        df2['moldura'] = df2[bola_cols].apply(lambda row: len(set(row) & MOLDURA), axis=1)
        features = ['soma', 'pares', 'repetidas', 'primos', 'moldura']
        feat_df = df2[features].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feat_df)
        kmeans = KMeans(n_clusters=n_perfis, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled)
        centros = scaler.inverse_transform(kmeans.cluster_centers_)
        perfis = []
        for i, centro in enumerate(centros):
            perfis.append({
                "Soma Média": int(round(centro[0])),
                "Pares Média": int(round(centro[1])),
                "Repetidas Média": int(round(centro[2])),
                "Primos Média": int(round(centro[3])),
                "Moldura Média": int(round(centro[4])),
                "Ocorrências": int((labels == i).sum())
            })
        return perfis, pd.Series(labels).value_counts()

    # backtest silencioso para otimização (mantido, usando avaliar_jogo)
    def _backtest_silencioso(self, pesos, alvo):
        num_concursos = int(self.config.get('NUMERO_DE_CONCURSOS_A_TESTAR', 50))
        concursos_para_testar = self.df.tail(num_concursos)
        custo_total = 0.0
        retorno_total = 0.0
        estrategia_teste = {"pesos": pesos, "alvo": alvo}
        for _, concurso in concursos_para_testar.iterrows():
            num_concurso = int(concurso['Concurso'])
            df_hist = self.df[self.df['Concurso'] < num_concurso]
            if df_hist.empty:
                continue
            resultado_real = sorted([int(concurso[f'Bola{i}']) for i in range(1, 16)])
            dados = {
                'concurso_anterior': sorted([int(df_hist.iloc[-1][f'Bola{i}']) for i in range(1, 16)]),
                'quentes': obter_dezenas_quentes(df_hist, DEFAULT_NUM_QUENTES),
                'frias': obter_dezenas_frias(df_hist, DEFAULT_NUM_FRIAS)
            }
            jogos = gerar_melhores_jogos(self.config.get('JOGOS_POR_SIMULACAO', 5), dados, estrategia_teste,
                                        num_candidatos=self.config.get('NUM_CANDIDATOS', 2000))
            for jogo, score in jogos:
                custo_total += CUSTO_JOGO_15_DEZENAS
                acertos = len(set(jogo) & set(resultado_real))
                retorno_total += PREMIOS_FIXOS.get(acertos, 0.0)
        roi = ((retorno_total - custo_total) / custo_total * 100) if custo_total > 0 else -100
        return roi

    def otimizar_estrategia(self, perfil_alvo, status_placeholder):
        search_space = self.search_spaces.get("Descoberto")
        resultados_parciais = []

        @use_named_args(search_space)
        def func_obj(**params):
            roi = self._backtest_silencioso(params, perfil_alvo)
            resultados_parciais.append(roi)
            iteracao_atual = len(resultados_parciais)
            mensagem = f"Teste {iteracao_atual}/{self.config.get('OTIMIZACAO_CHAMADAS', 20)} -> ROI: {roi:.2f}%"
            try:
                with status_placeholder:
                    st.write(mensagem)
            except Exception:
                print(mensagem)
            return -roi

        resultado = gp_minimize(func_obj, dimensions=search_space, n_calls=self.config.get('OTIMIZACAO_CHAMADAS', 20), random_state=42)
        melhor_roi = -resultado.fun
        melhores_pesos = {dim.name: int(val) for dim, val in zip(search_space, resultado.x)}
        return {"roi": melhor_roi, "pesos": melhores_pesos}

# -----------------------------
# FUNÇÕES DE EXPORTAÇÃO E UI AUX
# -----------------------------
def gerar_graficos(resultados_por_estrategia, historico_lucro):
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    nomes = list(resultados_por_estrategia.keys())
    lucros = [v['retorno'] - v['custo'] for v in resultados_por_estrategia.values()]
    cores = ['royalblue' if l >= 0 else 'salmon' for l in lucros]
    bars = ax1.bar(nomes, lucros, color=cores)
    ax1.set_ylabel('Lucro / Prejuízo (R$)')
    ax1.set_title('Resultado Financeiro Final por Estratégia')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.bar_label(bars, fmt='R$ %.2f')
    plt.xticks(rotation=45, ha='right')

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(historico_lucro, marker='o', linestyle='-')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.2)
    ax2.set_title('Evolução do Saldo Financeiro Acumulado')
    ax2.set_xlabel('Concursos Simulados')
    ax2.set_ylabel('Saldo Acumulado (R$)')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    return fig1, fig2

def conferir_jogos(texto_resultado: str, texto_apostas: str):
    nums = list(map(int, re.findall(r'\d+', texto_resultado)))
    if len(nums) != 15:
        st.error(f"Resultado deve ter 15 dezenas — você forneceu {len(nums)}.")
        return None
    numeros_resultado = set(nums)
    apostas = []
    for i, linha in enumerate(texto_apostas.strip().splitlines()):
        encontrados = list(map(int, re.findall(r'\d+', linha)))
        if len(encontrados) == 15:
            apostas.append(sorted(encontrados))
        else:
            st.warning(f"Linha {i+1} ignorada: contém {len(encontrados)} dezenas (esperado 15).")
    if not apostas:
        st.error("Nenhuma aposta válida encontrada.")
        return None
    detalhes = []
    resumo = {i: 0 for i in range(11, 16)}
    retorno_total = 0.0
    for aposta in apostas:
        acertos = len(set(aposta) & numeros_resultado)
        premio = PREMIOS_FIXOS.get(acertos, 0.0)
        retorno_total += premio
        if 11 <= acertos <= 15:
            resumo[acertos] += 1
        detalhes.append({"Jogo": ", ".join(map(str, aposta)), "Acertos": acertos, "Prêmio (R$)": f"{premio:.2f}"})
    custo_total = len(apostas) * CUSTO_JOGO_15_DEZENAS
    lucro = retorno_total - custo_total
    return {"custo": custo_total, "retorno": retorno_total, "lucro": lucro, "resumo_acertos": resumo, "detalhes": pd.DataFrame(detalhes)}

# -----------------------------
# INTERFACE STREAMLIT (principal)
# -----------------------------
st.set_page_config(layout="wide", page_title="Analisador Lotofácil - Melhorado")
st.title("🤖 Analisador Lotofácil — Versão Otimizada")

# carregamento de dados e modelo
try:
    df = carregar_dados_google_sheets()
    model = configurar_modelo_ia()
    st.sidebar.success(f"Base carregada — último concurso: {df['Concurso'].max()}")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar dados: {e}")
    df = pd.DataFrame()
    model = None

modo = st.sidebar.radio("Modo:", ["Gerador de Jogos", "Conferidor de Apostas", "Simulador de Estratégias", "Painel de Estratégias", "Laboratório de IA"])

if modo == "Gerador de Jogos":
    st.header("Gerador de Jogos")
    estrategias = carregar_estrategias()
    nomes = [e['nome'] for e in estrategias]
    escolhido = st.selectbox("Estratégia:", nomes)
    jogos_qtd = st.slider("Quantos jogos gerar?", 1, 30, 6)
    if st.button("Gerar"):
        estrategia_obj = next((x for x in estrategias if x['nome'] == escolhido), None)
        if not estrategia_obj:
            st.error("Estratégia não encontrada.")
        else:
            CONFIG = {"NUM_CANDIDATOS": 20000}
            analisador = LotofacilAnalisador(df, CONFIG, model)
            with st.spinner("Gerando..."):
                res = analisador.gerar_jogos_com_analise(estrategia_obj, jogos_qtd)
            st.subheader("Jogos Gerados")
            df_jogos = pd.DataFrame([j for j, s in res['jogos']], columns=[f'D{i+1}' for i in range(15)])
            st.dataframe(df_jogos, use_container_width=True)
            st.download_button("Exportar CSV", df_jogos.to_csv(index=False).encode('utf-8'), file_name="jogos_gerados.csv")
            with st.expander("Análise IA / Fallback"):
                st.text(res['analise'])

elif modo == "Conferidor de Apostas":
    st.header("Conferidor")
    col1, col2 = st.columns(2)
    with col1:
        texto_resultado = st.text_area("Resultado oficial (15 dezenas)", value="", placeholder="1,2,3,4,...")
    with col2:
        texto_apostas = st.text_area("Seus jogos (um por linha)", value="", height=250)
    if st.button("Conferir"):
        if not texto_resultado or not texto_apostas:
            st.warning("Preencha ambos os campos.")
        else:
            out = conferir_jogos(texto_resultado, texto_apostas)
            if out:
                k1, k2, k3 = st.columns(3)
                k1.metric("Lucro / Prejuízo", f"R$ {out['lucro']:.2f}")
                k2.metric("Custo Total", f"R$ {out['custo']:.2f}")
                k3.metric("Retorno Total", f"R$ {out['retorno']:.2f}")
                st.dataframe(out['detalhes'], use_container_width=True)
                csv = out['detalhes'].to_csv(index=False).encode('utf-8')
                st.download_button("Exportar detalhes CSV", csv, file_name="detalhes_apostas.csv")

elif modo == "Simulador de Estratégias":
    st.header("Simulador / Backtest")
    concursos = st.sidebar.slider("Quantos concursos testar", 10, 500, 50, step=10)
    jogos_por_sim = st.sidebar.slider("Jogos por concurso (total)", 1, 50, 10)
    if st.sidebar.button("Rodar Simulação"):
        estrategias = carregar_estrategias()
        CONFIG = {"NUMERO_DE_CONCURSOS_A_TESTAR": concursos, "JOGOS_POR_SIMULACAO": jogos_por_sim, "NUM_CANDIDATOS": 10000, "ESTRATEGIAS_ATUAIS": estrategias}
        analisador = LotofacilAnalisador(df, CONFIG)
        with st.spinner("Rodando simulação..."):
            resultados = analisador.rodar_simulacao()
        m = resultados['metricas_gerais']
        k1, k2, k3 = st.columns(3)
        k1.metric("Lucro / Prejuízo Total", f"R$ {m['lucro_total']:.2f}", f"{m['roi_total']:.2f}% ROI")
        k2.metric("Custo Total", f"R$ {m['custo_total']:.2f}")
        k3.metric("Retorno Total", f"R$ {m['retorno_total']:.2f}")
        st.subheader("Desempenho por Estratégia")
        for nome, dado in resultados['resultados_por_estrategia'].items():
            lucro_est = dado['retorno'] - dado['custo']
            roi_est = (lucro_est / dado['custo'] * 100) if dado['custo'] > 0 else 0.0
            st.markdown(f"**{nome}** — Lucro: R$ {lucro_est:.2f} | ROI: {roi_est:.2f}%")
            df_ac = pd.DataFrame(list(dado['resumo_acertos'].items()), columns=['Acertos', 'Quant']).query("Quant > 0")
            if not df_ac.empty:
                st.dataframe(df_ac.set_index('Acertos'))
        fig1, fig2 = gerar_graficos(resultados['resultados_por_estrategia'], resultados['historico_lucro'])
        st.pyplot(fig1)
        st.pyplot(fig2)

elif modo == "Painel de Estratégias":
    st.header("Painel de Estratégias")
    estrategias = carregar_estrategias()
    if not estrategias:
        st.info("Nenhuma estratégia salva.")
    else:
        nomes = [e['nome'] for e in estrategias]
        sel = st.selectbox("Selecione", nomes)
        est = next((x for x in estrategias if x['nome'] == sel), None)
        if est:
            st.json(est)
            if st.button("Excluir estratégia"):
                estrategias.remove(est)
                salvar_estrategias(estrategias)
                st.success("Estratégia excluída.")
                st.experimental_rerun()
            if st.button("Rodar simulação (100 concursos)"):
                CONFIG = {"NUMERO_DE_CONCURSOS_A_TESTAR": 100, "JOGOS_POR_SIMULACAO": 10, "NUM_CANDIDATOS": 5000, "ESTRATEGIAS_ATUAIS": [est]}
                analisador = LotofacilAnalisador(df, CONFIG)
                with st.spinner("Executando..."):
                    res = analisador.rodar_simulacao()
                res_est = res['resultados_por_estrategia'][est['nome']]
                lucro = res_est['retorno'] - res_est['custo']
                roi = (lucro / res_est['custo'] * 100) if res_est['custo'] > 0 else 0.0
                st.metric("Resultado (100 concursos)", f"R$ {lucro:.2f}", f"{roi:.2f}% ROI")
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(res['historico_lucro'], marker='o')
                ax.axhline(0, color='red', linestyle='--')
                st.pyplot(fig)

elif modo == "Laboratório de IA":
    st.header("Laboratório de IA (Descoberta e Otimização)")
    st.warning("Processo demorado. Use com cautela.")
    n_perfis = st.sidebar.number_input("Quantos perfis?", 2, 8, 3)
    chamadas = st.sidebar.slider("Qualidade otimização (chamadas por perfil)", 10, 200, 20)
    if st.sidebar.button("Iniciar"):
        CONFIG = {"NUMERO_DE_CONCURSOS_A_TESTAR": 100, "JOGOS_POR_SIMULACAO": 10, "NUM_CANDIDATOS": 5000, "OTIMIZACAO_CHAMADAS": chamadas}
        analisador = LotofacilAnalisador(df, CONFIG, model)
        with st.status("Iniciando...", expanded=True) as status:
            perfis, counts = analisador.descobrir_perfis(n_perfis)
            st.write("Perfis encontrados:")
            st.json(perfis)
            ests = carregar_estrategias()
            for i, perfil in enumerate(perfis):
                status.update(label=f"Otimizando perfil {i+1}/{len(perfis)}")
                resultado_opt = analisador.otimizar_estrategia(perfil, status)
                novo_roi = resultado_opt['roi']
                novos_pesos = resultado_opt['pesos']
                nome = f"Descoberto {i+1}: Soma {perfil['Soma Média']}"
                # valida e salva se melhor que o pior
                rois = [e['roi'] for e in ests] if ests else []
                pior = min(rois) if rois else -999
                if novo_roi > pior:
                    nova = {"nome": nome, "tipo": "Descoberto", "roi": novo_roi, "alvo": perfil, "pesos": novos_pesos}
                    if len(ests) >= 4:
                        pior_idx = int(np.argmin(np.array(rois)))
                        ests[pior_idx] = nova
                    else:
                        ests.append(nova)
                    salvar_estrategias(ests)
                    st.success(f"Estratégia '{nome}' salva com ROI {novo_roi:.2f}%")
                else:
                    st.warning(f"Estratégia '{nome}' descartada (ROI {novo_roi:.2f}%).")
            status.update(label="Concluído", state="complete")
        st.balloons()

# fim do arquivo
