import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

"""Ruta a la carpeta del proyecto, donde se encuentran todos los archivos y base de datos, a rellenar"""
RUTA_BASE = ""

"""Ruta a la carpeta de los audios de al base de datos, a rellenar"""
CARPETA_AUDIOS = os.path.join(RUTA_BASE, "", "")
"""Salida del archivo con la información de todas las características de todos los audios de la carpeta, a rellenar"""
SALIDA = os.path.join(RUTA_BASE, "")
MODELO = "facebook/wav2vec2-xls-r-300m"
SR, SNR_DB, LOTE = 16000, 36.0, 8

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
procesador = Wav2Vec2FeatureExtractor.from_pretrained(MODELO)
modelo = Wav2Vec2Model.from_pretrained(
    MODELO, use_safetensors=True).to(dispositivo).eval()


def cargar(ruta):
    """Devuelve la onda como tensor 1-D mono a 16 kHz."""
    onda, sr = torchaudio.load(ruta)
    if sr != SR:
        onda = torchaudio.transforms.Resample(sr, SR)(onda)
    if onda.shape[0] > 1:
        onda = onda.mean(dim=0, keepdim=True)
    return onda.squeeze(0)[:SR * 15]


def neutralizar(onda):
    """Inyecta ruido a SNR fijo para enmascarar el suelo original."""
    rms = onda.pow(2).mean().sqrt() + 1e-12
    ruido = torch.randn_like(onda)
    ruido = ruido / (ruido.pow(2).mean().sqrt() + 1e-12)
    return onda + ruido * rms * 10.0 ** (-SNR_DB / 20.0)


@torch.no_grad()
def vectorizar(ondas):
    """Lista de ondas -> matriz (B, 2048) con [media, desviacion tipica]."""
    entrada = procesador([o.numpy() for o in ondas], sampling_rate=SR,
                         return_tensors="pt", padding=True,
                         return_attention_mask=True)
    valores = entrada.input_values.to(dispositivo)
    mascara = entrada.attention_mask.to(dispositivo)

    tramas = modelo(valores, attention_mask=mascara).last_hidden_state
    # Mascara a nivel de trama: sin ella la media incluiria el relleno de los
    # audios cortos y diluiria su vector.
    m = modelo._get_feature_vector_attention_mask(
        tramas.shape[1], mascara).unsqueeze(-1).float()

    n = m.sum(dim=1).clamp(min=1.0)
    media = (tramas * m).sum(dim=1) / n
    desviacion = (((tramas - media.unsqueeze(1)) ** 2 * m).sum(dim=1) / n
                  ).clamp(min=1e-8).sqrt()
    return torch.cat([media, desviacion], dim=1).cpu().numpy()


def procesar(split):
    """Extrae un split completo y lo guarda como un unico .npz."""
    tabla = pd.read_csv(os.path.join(CARPETA_AUDIOS, "protocols",
                                     f"{split}_metadata.csv"))
    carpeta = os.path.join(CARPETA_AUDIOS, split) #Carpeta del split (train,test,val)
    vectores, lote = [], [] #Inicializacion de los dos vectores

    #Coge los fichero 1 por 1 y los vectoriza. Lee la onda y la guarda en lote[]. Cuando termina, reinicia lote[]
    for fichero in tabla.filename:
        lote.append(neutralizar(cargar(os.path.join(carpeta, fichero))))
        if len(lote) == LOTE:
            vectores.append(vectorizar(lote))
            lote = []
    if lote:
        vectores.append(vectorizar(lote))

    os.makedirs(SALIDA, exist_ok=True)
    np.savez_compressed(os.path.join(SALIDA, f"{split}.npz"),
                        X=np.concatenate(vectores),
                        y=tabla.label.to_numpy(dtype=np.int64),
                        filenames=tabla.filename.to_numpy())


for split in ("train", "val", "test"):
    procesar(split)