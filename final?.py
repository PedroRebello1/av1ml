# Bloco 1: Importações e Configurações Iniciais
# =================================================

import requests
import time
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configurações visuais para os gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# Bloco 2: Coleta de Dados via API e Criação do DataFrame
# =========================================================

def coletar_dados_api(total_paginas=6):
    """
    Coleta dados de produtos de múltiplas páginas da API da KaBuM!.
    """
    base_url = "https://servicespub.prod.api.aws.grupokabum.com.br/catalog/v2/products-by-category/hardware/ssd-2-5"
    parametros = {
        'page_size': 100,
        'facet_filters': '',
        'sort': 'most_searched',
        'is_prime': 'false',
        'payload_data': 'products_category_filters',
        'include': 'gift'
    }
    todos_produtos = []

    print(f"Iniciando coleta de dados de {total_paginas} páginas da API...")

    for pagina in range(1, total_paginas + 1):
        parametros['page_number'] = pagina
        try:
            # Faz a requisição para a API com um timeout de 10 segundos
            response = requests.get(base_url, params=parametros, timeout=10)
            
            if response.status_code == 200:
                dados_pagina = response.json()
                produtos_da_pagina = dados_pagina.get('data', [])
                if not produtos_da_pagina:
                    print(f"Página {pagina} não retornou produtos. Encerrando coleta.")
                    break 
                todos_produtos.extend(produtos_da_pagina)
                print(f"Página {pagina} coletada com sucesso. ({len(produtos_da_pagina)} produtos)")
            else:
                print(f"Falha ao coletar dados da página {pagina}. Status: {response.status_code}")

        except requests.RequestException as e:
            print(f"Ocorreu um erro de conexão na página {pagina}: {e}")
        
        # Pausa de meio segundo para não sobrecarregar a API
        time.sleep(0.5)

    print(f"\nColeta finalizada. Total de {len(todos_produtos)} produtos brutos coletados.")
    return todos_produtos

def criar_dataframe_ssd(lista_produtos):
    """
    Converte a lista de produtos (dicionários) em um DataFrame do Pandas.
    """
    if not lista_produtos:
        return pd.DataFrame()
        
    dados_para_df = []
    for produto in lista_produtos:
        atributos = produto.get('attributes', {})
        fabricante = atributos.get('manufacturer', {})
        
        # Acessa o preço, priorizando o de oferta, se existir
        offer = atributos.get('offer')
        preco_final = offer.get('price_with_discount') if offer else atributos.get('price_with_discount')

        dados_para_df.append({
            'id_produto': produto.get('id'),
            'titulo': atributos.get('title'),
            'peso_gramas': atributos.get('weight'),
            'preco': preco_final,
            'openbox': atributos.get('is_openbox'),
            'avaliacao_media': atributos.get('score_of_ratings'),
            'id_fabricante': fabricante.get('id'),
            'nome_fabricante': fabricante.get('name')
        })
    return pd.DataFrame(dados_para_df)

# --- Execução da Coleta e Criação do DataFrame ---
produtos_brutos = coletar_dados_api(total_paginas=6)
if produtos_brutos:
    df_ssds_inicial = criar_dataframe_ssd(produtos_brutos)
    print("\nVisualização inicial do DataFrame:")
    print(df_ssds_inicial.head())


# Bloco 3: Análise Exploratória, Visualizações e Limpeza
# =======================================================

def analise_exploratoria_e_limpeza(df):
    """
    Realiza a análise exploratória e a limpeza inicial do DataFrame.
    """
    if df.empty:
        print("DataFrame para análise está vazio. Abortando.")
        return pd.DataFrame(), 0

    print("\n--- INICIANDO ANÁLISE EXPLORATÓRIA ---")
    print("\nResumo Estatístico (Antes da Limpeza):")
    print(df.describe())

    # Histograma de Preços
    sns.histplot(df['preco'], bins=40, kde=True).set(title='Distribuição de Preços dos SSDs (Antes da Limpeza)', xlabel='Preço (R$)', ylabel='Frequência')
    plt.show()

    # Top 10 Fabricantes
    plt.figure(figsize=(12, 6))
    df['nome_fabricante'].value_counts().nlargest(10).plot(kind='bar', color='skyblue').set(title='Top 10 Fabricantes por Quantidade de Produtos', xlabel='Fabricante', ylabel='Contagem')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Limpeza de outliers de preço e preenchimento de valores nulos
    preco_maximo = 3500
    n_antes = len(df)
    df_filtrado = df[df['preco'] <= preco_maximo].copy()
    n_depois = len(df_filtrado)
    print(f"\nRemovidos {n_antes - n_depois} outliers de preço (acima de R$ {preco_maximo}).")

    mediana_avaliacao = df_filtrado['avaliacao_media'].median()
    df_filtrado['avaliacao_media'].fillna(mediana_avaliacao, inplace=True)

    return df_filtrado, preco_maximo

# --- Execução da Análise ---
df_ssds_analisado, PRECO_MAXIMO_GLOBAL = analise_exploratoria_e_limpeza(df_ssds_inicial)


# Bloco 4: Engenharia de Features a Partir do Título
# ====================================================

def extrair_features_titulo(df):
    """
    Cria novas colunas ('features') a partir da coluna 'titulo'.
    """
    print("\n--- INICIANDO ENGENHARIA DE FEATURES ---")
    
    def extrair_capacidade_gb(titulo):
        titulo_str = str(titulo).lower()
        tb_match = re.search(r'(\d+)\s*tb', titulo_str)
        gb_match = re.search(r'(\d+)\s*gb', titulo_str)
        if tb_match: return int(tb_match.group(1)) * 1000
        if gb_match: return int(gb_match.group(1))
        return 0

    def extrair_tipo_interface_ssd(titulo):
        titulo_str = str(titulo).lower()
        if 'externo' in titulo_str: return 'Externo'
        if 'nvme' in titulo_str: return 'NVMe'
        if 'sata' in titulo_str: return 'SATA'
        return 'Não especificado'
        
    def extrair_velocidade_leitura(titulo):
        titulo_str = str(titulo).lower()
        # Padrão para encontrar "leitura" seguido por um número (Ex: "leitura: 500 mb/s")
        match = re.search(r'leitura\s*[:\s]*(\d+)', titulo_str)
        return int(match.group(1)) if match else 0

    def extrair_velocidade_gravacao(titulo):
        titulo_str = str(titulo).lower()
        # Padrão para encontrar "gravação" (com ou sem cedilha) seguido por número
        match = re.search(r'grava[çc][ãa]o\s*[:\s]*(\d+)', titulo_str)
        return int(match.group(1)) if match else 0

    df['capacidade_gb'] = df['titulo'].apply(extrair_capacidade_gb)
    df['tipo_interface'] = df['titulo'].apply(extrair_tipo_interface_ssd)
    df['velocidade_leitura_mbps'] = df['titulo'].apply(extrair_velocidade_leitura)
    df['velocidade_gravacao_mbps'] = df['titulo'].apply(extrair_velocidade_gravacao)
    
    # Preenche velocidades não encontradas (valor 0) com a mediana das velocidades encontradas
    mediana_leitura = df[df['velocidade_leitura_mbps'] > 0]['velocidade_leitura_mbps'].median()
    mediana_gravacao = df[df['velocidade_gravacao_mbps'] > 0]['velocidade_gravacao_mbps'].median()
    df['velocidade_leitura_mbps'].replace(0, mediana_leitura, inplace=True)
    df['velocidade_gravacao_mbps'].replace(0, mediana_gravacao, inplace=True)
    df['velocidade_leitura_mbps'].fillna(mediana_leitura, inplace=True)
    df['velocidade_gravacao_mbps'].fillna(mediana_gravacao, inplace=True)


    # Remove produtos que não foi possível extrair a capacidade
    df_com_capacidade = df[df['capacidade_gb'] > 0].copy()
    print(f"{len(df) - len(df_com_capacidade)} produtos removidos por não ter sido possível extrair a capacidade do título.")
    
    # Filtra outliers de capacidade
    capacidade_maxima = 7000
    df_final = df_com_capacidade[df_com_capacidade['capacidade_gb'] <= capacidade_maxima].copy()
    print(f"{len(df_com_capacidade) - len(df_final)} outliers de capacidade removidos (acima de {capacidade_maxima} GB).")

    return df_final

# --- Execução da Engenharia de Features ---
df_ssds_com_features = extrair_features_titulo(df_ssds_analisado)

# Visualização Pós-Features
sns.scatterplot(data=df_ssds_com_features, x='capacidade_gb', y='preco', hue='tipo_interface', palette='viridis', s=100, alpha=0.7).set(title='Preço vs. Capacidade por Tipo de Interface', xlabel='Capacidade (GB)', ylabel='Preço (R$)')
plt.legend(title='Tipo de Interface')
plt.show()


# Bloco 5: Treinamento e Avaliação do Modelo Final (Regressão Polinomial com Log Transform)
# ========================================================================================

def treinar_modelo_final(df, preco_maximo):
    """
    Treina e avalia o modelo de Regressão Polinomial (Ridge) com transformação
    logarítmica no preço e limita as previsões ao teto de preço.
    """
    if df.empty:
        print("DataFrame para treinamento está vazio. Abortando.")
        return

    print("\n--- INICIANDO TREINAMENTO DO MODELO FINAL (REGRESSÃO POLINOMIAL + LOG TRANSFORM) ---")

    features = ['capacidade_gb', 'peso_gramas', 'velocidade_leitura_mbps', 'velocidade_gravacao_mbps', 'tipo_interface', 'nome_fabricante']
    target = 'preco'

    # Converte variáveis categóricas em numéricas (One-Hot Encoding)
    df_processed = pd.get_dummies(df, columns=['tipo_interface', 'nome_fabricante'], drop_first=True)
    
    features_processed = [f for f in features if f not in ['tipo_interface', 'nome_fabricante']]
    features_processed.extend([col for col in df_processed.columns if 'tipo_interface_' in col or 'nome_fabricante_' in col])

    X = df_processed[features_processed]
    # Aplica a transformação logarítmica. Usamos np.log1p para lidar com preços de valor 0 (se houver).
    y = np.log1p(df_processed[target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Cria um pipeline para automatizar as etapas: criar features, padronizar e treinar
    model_pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=10) # Alpha controla a regularização, bom para modelos complexos
    )

    model_pipeline.fit(X_train, y_train)
    # As previsões também estarão na escala logarítmica
    y_pred_log = model_pipeline.predict(X_test)

    # Reverte a transformação para avaliar o erro na escala original (R$)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_log)
    
    # **NOVA ETAPA: Limita as previsões ao teto de preço definido na limpeza**
    y_pred_clipped = np.clip(y_pred_orig, a_min=0, a_max=preco_maximo)
    print(f"\nAs previsões do modelo foram limitadas a um valor máximo de R$ {preco_maximo:.2f}.")
    
    # Avaliação do modelo com as previsões limitadas
    mse = mean_squared_error(y_test_orig, y_pred_clipped)
    mae = mean_absolute_error(y_test_orig, y_pred_clipped)
    r2 = r2_score(y_test_orig, y_pred_clipped)

    print("\nResultados da Avaliação do Modelo Final (com Previsões Limitadas):")
    print(f"  - Erro Quadrático Médio (MSE): {mse:.2f}")
    print(f"  - Erro Absoluto Médio (MAE): {mae:.2f}")
    print(f"  - Coeficiente de Determinação (R²): {r2:.2f}")

    # Visualização dos Resultados
    plt.figure(figsize=(12, 7))
    plt.scatter(y_test_orig, y_pred_clipped, alpha=0.7, edgecolors='k', label='Previsões vs. Real')
    perfect_line = np.linspace(min(y_test_orig.min(), y_pred_clipped.min()), max(y_test_orig.max(), y_pred_clipped.max()), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--', lw=2, label='Previsão Perfeita (R²=1.0)')
    plt.title('Preços Reais vs. Previsões (Modelo Polinomial com Log Transform)', fontsize=16)
    plt.xlabel('Preço Real (R$)', fontsize=12)
    plt.ylabel('Preço Previsto (R$)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return df

# --- Execução do Treinamento Final ---
df_ssds_final = treinar_modelo_final(df_ssds_com_features, PRECO_MAXIMO_GLOBAL)


# Bloco 6: Salvar o Dataset Final em CSV
# ========================================

if not df_ssds_final.empty:
    try:
        df_ssds_final.to_csv('ssds_dataset_final.csv', index=False)
        print("\nO DataFrame final foi salvo com sucesso em 'ssds_dataset_final.csv'")
    except Exception as e:
        print(f"\nOcorreu um erro ao salvar o arquivo: {e}")
else:
    print("\nDataFrame final não foi gerado, nada para salvar.")

