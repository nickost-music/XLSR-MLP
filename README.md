# XLSR-MLP Spanish
An antispoofing audio detector that uses representations extracted from XLS-R 300M model and uses a MLP trained on the HISPASPoof dataset.
---

## Descripción:

El sistema divide el proceso de detección en dos fases:
1. **Extracción de Características (Front-end):** Utiliza el modelo preentrenado `facebook/wav2vec2-xls-r-300m`. A cada audio se le aplica una neutralización de fondo (inyección de ruido a SNR fijo de 36 dB) para evitar sesgos de grabación, y se genera un vector de **2048 dimensiones** mediante la concatenación de la media y la desviación típica a nivel de trama considerando la máscara de atención.
2. **Clasificación (Back-end):** Una red neuronal MLP compacta de ~525K parámetros (`2048 -> 256 -> 2`), con normalización (`BatchNorm1d`), activación `LeakyReLU`, regularización por `Dropout(0.3)` y función de pérdida ponderada para manejar el desbalanceo de clases.
3. **Calibración de Umbral:** El punto de operación óptimo (*Equal Error Rate* o EER) se calibra sobre el conjunto de **validación** para evitar sesgos optimistas.

---

## Tree of documents:

```text
.
├── extract_features.py   # Extracción de embeddings XLS-R y guardado en .npz
├── train_mlp.py          # Entrenamiento del MLP y calibración de umbral
├── requirements.txt      # Dependencias del proyecto
├── pesos/
│   └── modelo.pth        # Pesos entrenados, media/desv de normalización y umbral (~2 MB)
└── README.md
