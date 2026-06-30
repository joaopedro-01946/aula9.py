import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configuração da página do Streamlit
st.set_page_config(page_title="Detector & Segmentador YOLOv8", layout="centered")

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo YOLOv8 nano voltado para segmentação."""
    return YOLO("yolov8n-seg.pt")

st.title("🔍 Identificação e Segmentação de Imagens")
st.write("Faça o upload de uma imagem para que o modelo YOLOv8 realize a segmentação dos objetos.")

# Componente de upload de arquivos
arquivo_carregado = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if arquivo_carregado is not None:
    # Converter o arquivo enviado para uma imagem PIL
    imagem_original = Image.open(arquivo_carregado)
    
    # Layout em colunas para comparação
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(imagem_original, use_container_width=True)
        
    # Inicializar o modelo
    modelo = carregar_modelo()
    
    with st.spinner("Processando e segmentando a imagem..."):
        # Executar a inferência do YOLOv8 diretamente na imagem PIL
        resultados = modelo(imagem_original)
        
        # Renderizar os resultados (máscaras e caixas) na imagem
        imagem_processada_bgr = resultados[0].plot()
        
        # Converter de BGR (padrão OpenCV/YOLO) para RGB (padrão Streamlit/PIL)
        imagem_processada_rgb = cv2.cvtColor(imagem_processada_bgr, cv2.COLOR_BGR2RGB)
        
    with col2:
        st.subheader("Objetos Identificados")
        st.image(imagem_processada_rgb, use_container_width=True)