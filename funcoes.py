
# Bibliotecas necessárias

import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
from geopy.distance import geodesic as GD
import folium
from scipy.spatial import distance_matrix
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d




################## Histograma com boxplot ###################
def hist_boxplot(df, var):

    """
    Retorna o gráfico de distribuição da variável

    Parâmentros
    -----------
    df: dataframe
    var: variavel numérica
    """

    numerical_var = df[var]
    sns.set(style="ticks")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(numerical_var , ax=ax_box)
    sns.distplot(numerical_var, ax=ax_hist)

    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)

################ Tabela resumo #####################

def descritiva(var,df,precisao = 2,drop_na = False,mult = 1):
    """
    Retorna a visualização da distribuição na tabela

    Parâmetros
    -----------
    var: variavel que queremos a distribuição
    df: Dataframe
    """

    resumo = {'n': df.shape[0],
    'na' : df[var].isna().sum(),
    'na (%)' : round(df[var].isna().sum()/df.shape[0], precisao),
    'média' : round(df[var].mean()*mult,precisao),
    'min' : round(df[var].min()*mult,precisao),
    '10%' : round(df[var].quantile(0.10)*mult,precisao),
    '20%' : round(df[var].quantile(0.20)*mult,precisao),
    '30%' : round(df[var].quantile(0.30)*mult,precisao),
    '40%' : round(df[var].quantile(0.40)*mult,precisao),
    'mediana' : round(df[var].median()*mult,precisao),
    '60%' : round(df[var].quantile(0.60)*mult,precisao),
    '70%' : round(df[var].quantile(0.70)*mult,precisao),
    '80%' : round(df[var].quantile(0.80)*mult,precisao),
    '90%' : round(df[var].quantile(0.90)*mult,precisao),
    'max' : round(df[var].max()*mult,precisao),
    'desvio' : round(df[var].std()*mult,precisao)}
    return resumo


################ Função de categorização do preço #####################

def categoriza_preco(price):
    """
    Retorna o campo de preco categorizado

    Parâmetros
    -----------
    price: campo de preço
    """

    if pd.isna(price):
        return np.nan
    elif price <= 100:
        return '0 - 100'
    elif price > 100 and price <= 200:
        return '101 - 200'
    elif price > 200 and price <= 300:
        return '201 a 300'
    elif price > 300 and price <= 400:
        return '301 a 400'
    else:
        return 'Mais que 401'
    
############# Tabela de frequencia relativa ###############

def tabela_frequencia_relativa(df, var1, var2):
    """
    Retorna a tabela de frequancia relativa entre duas variáveis categóricas

    Parâmetros
    -----------
    df : dataframe
    var1: variável categórica 1
    var2: variável categórica 2
    """

    my_crosstab = pd.crosstab(index=df[var1], 
                                columns=df[var2],
                                margins=True)   # Include row and column totals
    my_crosstab_prop = round(my_crosstab.div(my_crosstab["All"],axis=0)*100,2)
    my_crosstab_prop.reset_index(inplace=True)
    my_crosstab_prop = my_crosstab_prop.rename_axis(None, axis=1)
    my_crosstab_prop= my_crosstab_prop.set_index(var1)
    my_crosstab_prop.rename(columns={var1: 'Preço($)', var2: ' ', 'All': 'Total (%)'}, index={'All': 'Total (%)'}, inplace = True)
    my_crosstab_prop.index.name = 'Preço ($)'
    return my_crosstab_prop

######### Boxplot #################

def boxplot_categorica(df, var1, var2, title, xlabel, ylabel):
    """
    Retorna o gráfico boxplot

    Parâmetros
    -----------
    df: dataframe
    var1: variável numérica
    var2: variável categórica
    title: Título do gráfico
    xlabel: Título do eixo x
    ylabel: Título do eixo y
    """

    g = (ggplot(df[df[var1] < round(df[var1].quantile(0.99))], aes(x=var2, y = var1))
        + geom_boxplot(fill = '#0069F8')
        + labs(
                title = title,
                y = ylabel,
                x = xlabel)
        + theme_classic() 
        + theme(axis_text_x=element_text(rotation=25, hjust=1)) 
    )
    return print(g)

######### Tabela descritiva por agruoamento ############
def tabela_descritiva_agrupamento(var1,var2,df,precisao = 2,drop_na = False,mult = 1):
    """
    Retorna a visualização da distribuição na tabela

    Parâmentros
    -----------
    var1: variavel que queremos a distribuição
    var2: variavel de agrupamento
    df: dataframe em estamos trabalhando
    precisao: numero de casas decimais
    drop_na: "False" ou "True", se queremos dropna
    mult: se for taxa usar 100 para mutiplicar, se não usar 1
    """

    visu= pd.concat([round(df[var1].groupby(df[var2],dropna=drop_na).agg(['count'])),
    round(df[var1].groupby(df[var2],dropna=drop_na).agg(['count'])
    /df[var1].groupby(df[var2],dropna=drop_na).agg(['count']).sum()*100,2),
    round(df.groupby(var2,dropna=drop_na)[var1].mean()*mult,precisao),
    round(df.groupby(var2,dropna=drop_na)[var1].std()*mult,precisao),
    round(df.groupby(var2,dropna=drop_na)[var1].min()*mult,precisao),
    round(df.groupby(var2,dropna=drop_na)[var1].quantile(0.25)*mult,precisao),
    round(df.groupby(var2,dropna=drop_na)[var1].median()*mult,precisao),
    round(df.groupby(var2,dropna=drop_na)[var1].quantile(0.75)*mult,precisao),
    round(df.groupby(var2,dropna=drop_na)[var1].max()*mult,precisao)],
    axis=1, ignore_index = drop_na)
    columns = ['n', '%','Média','Desvio','Min',"25%",'Mediana',"75%",'Max']
    visu.columns = columns
    visu.sort_values('n', ascending = False, inplace = True)
    visu['% acumulada'] = visu['%'].cumsum()


    return visu[['n', '%','% acumulada','Média','Desvio','Min',"25%",'Mediana',"75%",'Max']]

####### Função de tradução #########

def traducao(df, coluna):
    """
    Retorna um dicionário com a tradução de cada texto presente na coluna

    Parâmentros
    -----------
    df: dataframe
    coluna: campo a ser traduzido
    """
    d = {}
    for i in df[coluna].unique():
        d[i] = GoogleTranslator(source='auto', target='pt').translate(i)
    return d

####### Função para tratar valores não numéricos na variável bathroom #########

def numerica(x):
    try:
        float(x)
        return x
    except:
        return np.nan

######## Função para corregir latitude e longitude ##########

def tratar_latitude(cols):
    """
    Retorna a latitude corrigida

    Parâmentros
    -----------
    cols: Campos latitude e seu tamanho
    """

    latitude = cols[0]
    latitude_len = cols[1]
    if latitude_len == 5:
        return latitude/1000
    elif latitude_len == 6:
        return latitude/10000
    elif latitude_len == 7:
        return latitude/100000
    else:
        return latitude
    
def tratar_longitude(cols):
    """
    Retorna a loongitude corrigida

    Parâmentros
    -----------
    cols: Campos longitude e seu tamanho
    """
    longitude = cols[0]
    longitude_len = cols[1]
    if longitude_len == 7:
        return longitude/1000
    elif longitude_len == 8:
        return longitude/10000
    elif longitude_len == 9:
        return longitude/100000
    else:
        return longitude
    
###### Função para o cálculo de distância #####
def calcula_distancia(geo_cols, geolocalizacao_ponto):
    """
    Retorna a distância (em km) entre a locação e algum ponto turístico

    Parâmentros
    -----------
    geo_cols: Latitude e longitude da locação
    geolocalizacao_ponto : Tupla com a latitude e longitude do ponto turístico
    """
    geolocalizacao_locacao = tuple([geo_cols[0], geo_cols[1]])
    return GD(geolocalizacao_locacao,geolocalizacao_ponto).km


###### Construção do mapa ########
def constroi_mapa(df):
    lat_mean = 37.75742096307778
    lng_mean = -122.47018734576127

    cluster_map = folium.Map(location=[lat_mean, lng_mean], zoom_start=10, tiles='cartodbpositron')

    colors = ['#f5c122','#cc0000','#3d85c6','#0b5394','#b400ff','#00a4ff', '#5faf3c', '#f27778','#f7239f','#41f723','#010600','#f56828','#b774f7','#ce7e00','#c90076','#999999','#444444','#8040c0', '#fa77ab', '#a6cbe8', '#8fdebe', '#007c7c', '#b38e57', '#562d0b', '#7d645b', '#a4c9ff', '#ff7ada', '#bcff3e', '#ed4e4e', '#438fff', '#8040c0', '#fa77ab', '#a6cbe8', '#8fdebe', '#007c7c', '#b38e57']
    for i, v in df.iterrows():
            popup = """
            Latitude : <b>%s</b><br>
            Longitude : <b>%s</b><br>
            Bairro : <b>%s</b><br>
            Região : <b>%s</b><br>
            Faixa de preço ($) : <b>%s</b><br>
            """ % (v['latitude_corrigida'], v['longitude_corrigida'], v['neighbourhood_cleansed'], v['regioes'], v['preco_predito_categoria'] )
            

            folium.CircleMarker(location=[v['latitude_corrigida'], v['longitude_corrigida']],
                            radius=2,
                            tooltip=popup,
                            color=colors[int(v['regioes'])%len(colors)],
                            fill_color=colors[int(v['regioes'])%len(colors)],
                            fill=True).add_to(cluster_map)

    popup1 = """
            Ponto turístico : <b>%s</b><br>
            """ % ('Alcatraz')
            
    folium.Marker(location=[37.827024820594126, -122.42275276614855],
                        tooltip=popup1,
                        icon=folium.Icon(color='green', icon="globe")).add_to(cluster_map)
    
    popup2 = """
            Ponto turístico : <b>%s</b><br>
            """ % ('Ponte Golden Gate ')
            
    folium.Marker(location=[37.820131983211965, -122.47859842468475],
                        tooltip=popup2,
                        icon=folium.Icon(color='green', icon="globe")).add_to(cluster_map)
    
    popup3 = """
            Ponto turístico : <b>%s</b><br>
            """ % ('Cable Cars')
            
    folium.Marker(location=[37.79477222857774, -122.4115610461312],
                        tooltip=popup3,
                        icon=folium.Icon(color='green', icon="globe")).add_to(cluster_map)
    
    return cluster_map