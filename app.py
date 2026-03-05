

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import textwrap
import io
import os

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILO
# ==========================================
st.set_page_config(
    page_title="Auditoría algorítmica de los desemboldos del BCIE para los países fundadores",
    page_icon="🏦",
    layout="wide"
)

# Estilo de gráficos
sns.set(style="whitegrid", context="talk")

st.title("🏦 El mapa oculto del desarrollo: cómo la cartera de desembolsos del BCIE está reconfigurando la región CA5")
st.markdown("""
**Auditoría algorítmica de datos abiertos:**
Esta aplicación conecta en tiempo real a las APIs del BCIE y del SDG Index, aplicando técnicas de vectorización semántica (S-BERT), minería de texto y aprendizaje no supervisado para auditar la estructura funcional y financiera de la cartera de proyectos de los países fundadores para el período 2010-2024.
""")
st.markdown("""
**Alfredo Ibrahim Flores Sarria © 2026** 
""")

# ==========================================
# 2. FUNCIONES DE CARGA (CON CACHÉ)
# ==========================================

@st.cache_resource
def cargar_modelo_sbert():
    """Carga el modelo S-BERT en memoria (solo una vez)."""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data(ttl=3600) # Caché por 1 hora
def descargar_datos_bcie():
    """Descarga, limpia y preprocesa los datos de la API del BCIE."""
    base_url = "https://datosabiertos.bcie.org/api/3/action/datastore_search"
    resource_id = "8794659a-285b-4f0d-b5a4-a704bdf823fb"
    params = {'resource_id': resource_id, 'limit': 50000}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data_json = response.json()

        if data_json['success']:
            df = pd.DataFrame(data_json['result']['records'])

            # Normalización
            df.columns = df.columns.str.upper().str.strip()

            # Tipos de datos
            if 'ANIO_DESEMBOLSO' in df.columns:
                df['ANIO_DESEMBOLSO'] = pd.to_numeric(df['ANIO_DESEMBOLSO'], errors='coerce')

            # Filtro Países y Años
            target_countries = ['GUATEMALA', 'EL SALVADOR', 'HONDURAS', 'NICARAGUA', 'COSTA RICA']
            if 'PAIS' in df.columns and 'ANIO_DESEMBOLSO' in df.columns:
                df['PAIS'] = df['PAIS'].str.upper().str.strip()
                df_filtered = df[
                    (df['PAIS'].isin(target_countries)) &
                    (df['ANIO_DESEMBOLSO'] >= 2010) &
                    (df['ANIO_DESEMBOLSO'] <= 2024)
                ].copy()

                # Limpieza de texto
                df_filtered['DESCRIPCION_PROYECTO'] = df_filtered['DESCRIPCION_PROYECTO'].fillna('').astype(str)
                df_filtered = df_filtered[df_filtered['DESCRIPCION_PROYECTO'].str.len() > 5]

                return df_filtered
            else:
                return None
        else:
            return None
    except Exception as e:
        st.error(f"Error conectando a API BCIE: {e}")
        return None

@st.cache_data(ttl=3600)
def cargar_datos_sdg():
    """Descarga los datos del SDG Index 2025 desde ArcGIS."""
    url = "https://services7.arcgis.com/IyvyFk20mB7Wpc95/arcgis/rest/services/Sustainable_Development_Report_2025_(with_indicators)/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'features' in data:
            rows = [feat['attributes'] for feat in data['features']]
            df = pd.DataFrame(rows)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error conectando a API SDG: {e}")
        return pd.DataFrame()

def extraer_palabras_clave(texts, n=5):
    """Extrae características clave usando TF-IDF."""
    tfidf = TfidfVectorizer(stop_words=['de', 'la', 'el', 'en', 'y', 'los', 'del', 'para', 'las', 'un', 'con', 'por', 'proyecto', 'programa'],
                            ngram_range=(2, 3))
    try:
        matrix = tfidf.fit_transform(texts)
        feature_names = np.array(tfidf.get_feature_names_out())
        tfidf_mean = matrix.mean(axis=0).A1
        top_indices = tfidf_mean.argsort()[::-1][:n]
        return feature_names[top_indices]
    except ValueError:
        return ["Datos insuficientes"]

# ==========================================
# 3. LÓGICA DE EJECUCIÓN
# ==========================================

# Botón principal para iniciar todo el proceso
if st.button('▶️ EJECUTAR AUDITORÍA COMPLETA', type="primary"):

    # --- PASO 1: INGESTA BCIE ---
    with st.status("📡 Conectando a fuentes de datos...", expanded=True) as status:
        st.write("Descargando registros del BCIE...")
        df_bcie = descargar_datos_bcie()

        if df_bcie is not None and not df_bcie.empty:
            st.write(f"✅ BCIE: {len(df_bcie)} registros procesados.")
        else:
            st.error("No se pudieron cargar los datos del BCIE.")
            st.stop()

        status.update(label="✅ Datos descargados", state="complete", expanded=False)

    # --- PASO 2: MOTOR DE ALGORITMOS ---
    with st.spinner("⚙️ Ejecutando algoritmos (S-BERT, K-Means++)..."):
        model = cargar_modelo_sbert()
        embeddings = model.encode(df_bcie['DESCRIPCION_PROYECTO'].tolist())

        # Clustering
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42, n_init=10)
        df_bcie['Cluster_ID'] = kmeans.fit_predict(embeddings)

        # Feature Extraction
        feature_map = {}
        for k in range(num_clusters):
            subset = df_bcie[df_bcie['Cluster_ID'] == k]['DESCRIPCION_PROYECTO']
            top_features = extraer_palabras_clave(subset, n=5)
            feature_map[k] = ", ".join(top_features)

    # --- PREPARACIÓN DE DATOS FINANCIEROS ---
    col_monto = 'MONTO_BRUTO_USD'
    if col_monto in df_bcie.columns:
        # Limpieza
        df_bcie[col_monto] = df_bcie[col_monto].astype(str).str.replace(r'[$,]', '', regex=True)
        df_bcie[col_monto] = pd.to_numeric(df_bcie[col_monto], errors='coerce').fillna(0)

        # Pivots
        pivot_monto = df_bcie.pivot_table(index='Cluster_ID', columns='PAIS', values=col_monto, aggfunc='sum').fillna(0)
        pivot_conteo = df_bcie.pivot_table(index='Cluster_ID', columns='PAIS', aggfunc='size').fillna(0)

        # Millones
        pivot_monto_millones = pivot_monto / 1_000_000

        # Valor promedio por operación
        valor_promedio = pivot_monto_millones / pivot_conteo

        # Renombrar índices para visualización
        etiquetas_legibles = [f"C{i}\n({feature_map[i][:25]}...)" for i in pivot_monto.index]
        pivot_monto_millones.index = etiquetas_legibles
        valor_promedio.index = etiquetas_legibles

    # --- INTERFAZ DE RESULTADOS (TABS) ---
    st.success("✅ Auditoría finalizada. Resultados listos.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌌 Espacio Latente de Operaciones",
        "💰 Distribución de Recursos por Clúster Semántico",
        "📏 Análisis de Escala: Valor Promedio por Operación",
        "🌍 Índice de los ODS 2025"
    ])

    # TAB 1: VISUALIZACIÓN ESPACIO LATENTE
    with tab1:
        st.subheader("Figura 1. Auditoría Algorítmica: Espacio Latente de Operaciones en los países fundadores del BCIE")
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Leyenda limpia
        etiquetas_plot = [f"C{k}: {feature_map[k][:30]}..." for k in df_bcie['Cluster_ID']]
        
        sns.scatterplot(
            x=coords[:, 0], y=coords[:, 1],
            hue=etiquetas_plot,
            style=df_bcie['PAIS'],
            palette='viridis', s=100, alpha=0.8, edgecolor='k', ax=ax1
        )
        ax1.set_xlabel("Dimensión Latente 1")
        ax1.set_ylabel("Dimensión Latente 2")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig1)
        
        

        # --- BOTÓN DE DESCARGA TAB 1 ---
        # Preparamos un DataFrame especial que incluya las coordenadas matemáticas
        df_export_pca = df_bcie.copy()
        df_export_pca['Dim_Latente_X'] = coords[:, 0]
        df_export_pca['Dim_Latente_Y'] = coords[:, 1]
        df_export_pca['Etiqueta_Cluster'] = etiquetas_plot
        
        buffer_pca = io.BytesIO()
        with pd.ExcelWriter(buffer_pca, engine='xlsxwriter') as writer:
            df_export_pca.to_excel(writer, sheet_name='Datos_Espacio_Latente', index=False)
            
        st.download_button(
            label="💾 Descargar Datos del Espacio Latente (Excel)",
            data=buffer_pca,
            file_name="BCIE_Espacio_Latente.xlsx",
            mime="application/vnd.ms-excel"
        )
        

    # TAB 2: MATRIZ DE DESEMBOLSOS
    with tab2:
        st.subheader("Figura 2. Distribución de Recursos por Clúster Semántico en los países fundadores del BCIE")
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # Heatmap Verde
        sns.heatmap(pivot_monto_millones, annot=True, fmt=".2f", cmap="Greens", linewidths=1, ax=ax2)
        ax2.set_ylabel("Clúster Semántico")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, ha='right', fontsize=11)
        st.pyplot(fig2)

        # Botón descarga
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_bcie.to_excel(writer, sheet_name='Data_Cruda', index=False)
            pivot_monto_millones.to_excel(writer, sheet_name='Matriz_Montos')

        st.download_button(
            label="💾 Descargar Resultados Financieros (Excel)",
            data=buffer,
            file_name="BCIE_Auditoria_Resultados.xlsx",
            mime="application/vnd.ms-excel"
        )

    
    # TAB 3: VALOR PROMEDIO POR OPERACIÓN
    with tab3:
        st.subheader("Figura 3. Análisis de Escala: Valor Promedio por Operación en los países fundadores del BCIE")
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        
        # Heatmap Azul
        sns.heatmap(valor_promedio.fillna(0), annot=True, fmt=".2f", cmap="Blues", linewidths=1, ax=ax3)
        ax3.set_ylabel("Clúster Semántico")
        
        # Rotación de etiquetas
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0, ha='right', fontsize=11)
        
        st.pyplot(fig3)
        


        # --- BOTÓN DE DESCARGA TAB 3 ---
        buffer_avg = io.BytesIO()
        with pd.ExcelWriter(buffer_avg, engine='xlsxwriter') as writer:
            df_bcie.to_excel(writer, sheet_name='Data_Cruda', index=False)
            valor_promedio.to_excel(writer, sheet_name='Matriz_Valor_Promedio')
        
        st.download_button(
            label="💾 Descargar Matriz de Escala (Excel)",
            data=buffer_avg,
            file_name="BCIE_Analisis_Escala.xlsx",
            mime="application/vnd.ms-excel"
        )

# TAB 4: CONTEXTO ODS (ArcGIS)
    with tab4:
        st.subheader("Figura 4. Países fundadores del BCIE: Índice de los ODS 2025")

        with st.spinner("Conectando a ArcGIS..."):
            df_sdg = cargar_datos_sdg()

            paises_bcie = ['Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua']
            if not df_sdg.empty and 'Name' in df_sdg.columns:
                df_sdg_filt = df_sdg[df_sdg['Name'].isin(paises_bcie)].copy()
                df_sdg_filt = df_sdg_filt.sort_values('Overall_Score', ascending=True)

                fig4, ax4 = plt.subplots(figsize=(10, 6))
                bars = ax4.barh(df_sdg_filt['Name'], df_sdg_filt['Overall_Score'], color='#0055A4')

                ax4.set_xlabel('Puntaje General (Overall Score)')
                ax4.grid(axis='x', linestyle='--', alpha=0.5)
                ax4.set_xlim(0, 90)

                # Etiquetas en barras
                for bar in bars:
                    width = bar.get_width()
                    ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                             f'{width:.1f}', va='center', fontweight='bold')

                st.pyplot(fig4)
                
                # --- NUEVA SECCIÓN: MOSTRAR DATOS Y BOTÓN DE DESCARGA ---
                st.markdown("### Datos extraídos:")
                
                # Filtramos las columnas de interés
                df_export_sdg = df_sdg_filt[['Name', 'Overall_Score', 'Overall_Rank']].copy()
                
                # Opcional: Mejorar el formato para la vista en pantalla y Excel
                df_export_sdg['Overall_Score'] = df_export_sdg['Overall_Score'].round(2)
                # Convertimos el ranking a número entero, manejando posibles valores nulos
                df_export_sdg['Overall_Rank'] = pd.to_numeric(df_export_sdg['Overall_Rank'], errors='coerce').fillna(0).astype(int)
                
                # Mostramos la tabla en la interfaz de Streamlit
                st.dataframe(df_export_sdg, use_container_width=True)

                # Preparamos el buffer para la descarga en Excel
                buffer_sdg = io.BytesIO()
                with pd.ExcelWriter(buffer_sdg, engine='xlsxwriter') as writer:
                    df_export_sdg.to_excel(writer, sheet_name='Indice_ODS_2025', index=False)
                
                st.download_button(
                    label="💾 Descargar Datos ODS (Excel)",
                    data=buffer_sdg,
                    file_name="BCIE_Indice_ODS_2025.xlsx",
                    mime="application/vnd.ms-excel"
                )

            else:
                st.warning("No se pudieron recuperar los datos del SDG Index en este momento.")
