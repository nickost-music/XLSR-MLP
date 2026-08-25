"""Entrenamiento del clasificador MLP sobre las caracteristicas neutralizadas.

Arquitectura:  2048 -> 256 -> 2   (525 570 parametros)
Etiquetas:     0 = real,  1 = sintetico

Al terminar se calcula el umbral de decision sobre el conjunto de VALIDACION
y se guarda en el checkpoint. Fijarlo con validacion, y no con el conjunto que
luego se reporta, evita el sesgo optimista de elegir el umbral que minimiza el
error en los propios datos evaluados.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve


RUTA_BASE = ""      # ruta raiz del proyecto
DIR_EMB = os.path.join(RUTA_BASE, "")      # carpeta de embeddings neutralizados
DIR_PESOS = os.path.join(RUTA_BASE, "")    # carpeta de salida de los pesos

EPOCAS = 7
LOTE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEMILLA = 1234

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClasificadorMLP(nn.Module):
    def __init__(self, dim_entrada=2048, dim_oculta=256, n_clases=2):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculta),
            nn.BatchNorm1d(dim_oculta),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(dim_oculta, n_clases),
        )

    def forward(self, x):
        return self.red(x)


def cargar(split):
    d = np.load(os.path.join(DIR_EMB, f"{split}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

X, y = cargar("train")

# Estadisticos calculados solo con train; se guardan en el checkpoint porque
# la inferencia debe aplicar exactamente la misma normalizacion.
media = X.mean(axis=0)
desv = X.std(axis=0) + 1e-8
X = torch.from_numpy((X - media) / desv).to(dispositivo)
y = torch.from_numpy(y).to(dispositivo)

# Pesos inversamente proporcionales a la frecuencia: el conjunto esta
# desbalanceado (~1:4) y sin esto la red tiende a la clase mayoritaria.
n_real = int((y == 0).sum())
n_ia = int((y == 1).sum())
pesos = torch.tensor([(n_real + n_ia) / (2 * n_real),
                      (n_real + n_ia) / (2 * n_ia)],
                     dtype=torch.float32, device=dispositivo)

modelo = ClasificadorMLP().to(dispositivo)
criterio = nn.CrossEntropyLoss(weight=pesos)
optimizador = torch.optim.Adam(modelo.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador,
                                                          T_max=EPOCAS)

for epoca in range(1, EPOCAS + 1):
    modelo.train()
    orden = torch.randperm(len(X), device=dispositivo)
    perdida_total = 0.0

    for i in range(0, len(orden), LOTE):
        idx = orden[i:i + LOTE]
        if len(idx) < 2:          # BatchNorm necesita al menos dos muestras
            continue
        optimizador.zero_grad()
        perdida = criterio(modelo(X[idx]), y[idx])
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item() * len(idx)

    planificador.step()
    print(f"epoca {epoca}/{EPOCAS}  perdida = {perdida_total / len(orden):.4f}")

# ---------------------------------------------------------------- calibracion
X_val, y_val = cargar("val")
X_val = torch.from_numpy((X_val - media) / desv).to(dispositivo)

modelo.eval()
with torch.no_grad():
    prob_val = torch.softmax(modelo(X_val), dim=1)[:, 1].cpu().numpy()

fpr, tpr, umbrales = roc_curve(y_val, prob_val)
fnr = 1 - tpr
i = int(np.nanargmin(np.abs(fnr - fpr)))
umbral = float(umbrales[i])
eer_val = (fpr[i] + fnr[i]) / 2 * 100

print(f"\nvalidacion: EER = {eer_val:.2f} %   umbral = {umbral:.6f}")

os.makedirs(DIR_PESOS, exist_ok=True)
torch.save({"modelo": modelo.state_dict(), "media": media, "desv": desv,
            "umbral": umbral, "eer_val": eer_val},
           os.path.join(DIR_PESOS, "modelo.pth"))