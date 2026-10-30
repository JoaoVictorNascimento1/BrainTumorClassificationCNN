import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# Imports necessários para o Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Dicionário de classes
CLASSES = {0: 'Glioma', 1: 'Meningioma', 2: 'Sem Tumor', 3: 'Tumor de Hipófise'}

@st.cache_resource
def carregar_modelo():
    # Instanciar o backbone EfficientNetB0
    modelo = models.efficientnet_b0(pretrained=False)
    
    # Recriar a cabeça exata usada no treinamento
    modelo.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 4)
    )
    
    # Carregar os pesos treinados
    modelo.load_state_dict(torch.load('diagnosAI_efficientnetb0.pth', map_location='cpu'))
    modelo.eval()
    return modelo

def preprocessar_imagem(imagem_pil):
    # Converter PIL para numpy para aplicar Otsu
    img_cv = np.array(imagem_pil)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # Aplicar Otsu e recorte de margens
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contornos:
        c = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img_crop = img_cv[y:y+h, x:x+w]
    else:
        img_crop = img_cv

    # Redimensionar a imagem recortada para 224x224
    img_crop_resized = cv2.resize(img_crop, (224, 224))
    
    # Converter para PIL e aplicar transformações do PyTorch para o modelo
    img_crop_pil = Image.fromarray(img_crop_resized)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_entrada = transform(img_crop_pil).unsqueeze(0)
    
    # Gerar a imagem base para o Grad-CAM (RGB, float32, escala de 0 a 1)
    img_base_cam = np.float32(img_crop_resized) / 255.0
    
    return tensor_entrada, img_base_cam

# Interface Streamlit
st.set_page_config(page_title="DiagnosAI", layout="centered")
st.title("DiagnoseAI 🧠")
st.write("Sistema de suporte à decisão clínica para classificação de tumores em MRI." \
"Este projeto serve apenas para propósitos educacionais.")

arquivo_upload = st.file_uploader("Faça o upload da ressonância magnética (JPG/PNG)", type=["jpg", "png", "jpeg"])

if arquivo_upload is not None:
    # Ler imagem
    imagem = Image.open(arquivo_upload).convert('RGB')
    
    if st.button("Analisar Exame", type="primary"):
        with st.spinner("Processando imagem e gerando explicações..."):
            modelo = carregar_modelo()
            tensor_entrada, img_base_cam = preprocessar_imagem(imagem)
            
            # Inferência
            with torch.no_grad():
                saida = modelo(tensor_entrada)
                probabilidades = torch.nn.functional.softmax(saida[0], dim=0)
                _, pred_idx = torch.max(saida, 1)
            
            classe_predita = CLASSES[pred_idx.item()]
            confianca = probabilidades[pred_idx.item()].item() * 100
            
            # Grad-CAM
            camadas_alvo = [modelo.features[-1]]
            
            # Criar o objeto GradCAM
            with GradCAM(model=modelo, target_layers=camadas_alvo) as cam:
                # O alvo é a classe com maior probabilidade
                targets = [ClassifierOutputTarget(pred_idx.item())]
                
                # Gerar a matriz do mapa de calor
                grayscale_cam = cam(input_tensor=tensor_entrada, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # Sobrepor o mapa de calor na imagem pré-processada
                mapa_de_calor_grad_cam = show_cam_on_image(img_base_cam, grayscale_cam, use_rgb=True)
            
            # Exibição dos Resultados
            st.success(f"**Diagnóstico Sugerido:** {classe_predita} (Confiança: {confianca:.2f}%)")
            
            # Organizar as imagens lado a lado para melhor comparação
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_base_cam, caption="Exame Recortado e Processado", use_container_width=True)
            with col2:
                st.image(mapa_de_calor_grad_cam, caption="Área de Maior Ativação (Grad-CAM)", use_container_width=True)
