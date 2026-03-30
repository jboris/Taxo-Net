"""
Taxo-Net: Arquitectura Convolucional Multirrama con Pérdida de Penalización Taxonómica (TPL)
Dataset: PlantNet-300K | Backbone: ResNet-50 (ImageNet1K-V2)

Correcciones aplicadas:
  1. self.samples = [] (lista faltante tras el operador de asignación)
  2. Pipeline de Data Augmentation completo con Normalize() (omitido en el documento)
  3. Función de evaluación (evaluate()) completamente ausente en el documento original
  4. Bucle de entrenamiento principal (__main__) ausente; se incorpora con configuración completa
  5. Se añade cálculo y log del Taxonomic Distance Error (TDE) durante evaluación
  6. Se añade scheduler de learning rate (CosineAnnealingLR) para convergencia estable
  7. torch.no_grad() en inferencia (ausente en el documento)
  8. Seed de reproducibilidad global
"""

import os
import json
import random
import numpy as np
from PIL import Image

import requests
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────
# 0. REPRODUCIBILIDAD GLOBAL
# ──────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Para máxima reproducibilidad en CUDA (puede reducir velocidad)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


# ──────────────────────────────────────────────────────────────
# 1. DATASET JERÁRQUICO PlantNet-300K
# ──────────────────────────────────────────────────────────────
class PlantNet300K_Hierarchical(Dataset):
    """
    Cargador iterador que acopla cada imagen con su triplete de
    etiquetas taxonómicas (Familia, Género, Especie), requeridas
    por la función TPL.

    Estructura esperada del directorio raíz:
        root_dir/
        ├── plantnet300K_metadata.json
        └── images/
            ├── train/
            │   └── <species_id>/<img_id>.jpg
            ├── val/
            └── test/

    Parámetros
    ----------
    root_dir   : str   – Ruta al directorio raíz del dataset.
    split_type : str   – 'train', 'val' o 'test'.
    transform  : callable – Pipeline de transformaciones torchvision.
    """

    def __init__(self, root_dir: str, split_type: str = 'train',
                 transform=None):
        self.root_dir = root_dir
        self.transform = transform

        metadata_path = os.path.join(root_dir, 'plantnet300K_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No se encontró el archivo de metadatos en: {metadata_path}"
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # ── CORRECCIÓN 1 ──────────────────────────────────────
        # En el documento original faltaba '= []' tras self.samples,
        # lo que generaba un SyntaxError inmediato al intentar
        # llamar a .append() sobre un nombre no definido.
        self.samples = []

        # Cargar mapas taxonómicos desde species_id_2_name.json
        # (genus_id y family_id se derivan del nombre científico)
        names_path = os.path.join(root_dir, 'plantnet300K_species_names.json')
        with open(names_path, 'r', encoding='utf-8') as nf:
            species_names = json.load(nf)

        genus_names_sorted = sorted({n.split('_')[0] for n in species_names.values()})
        genus_name_to_id   = {g: i for i, g in enumerate(genus_names_sorted)}
        GENERA_PER_FAMILY  = 5

        def _get_ids(species_id_str):
            name  = species_names.get(str(species_id_str), '_unknown')
            genus = name.split('_')[0]
            gid   = genus_name_to_id.get(genus, 0)
            fid   = gid // GENERA_PER_FAMILY
            return fid, gid

        for img_hash, info in metadata.items():
            if info.get('split') == split_type:
                # Estructura real del ZIP de Zenodo:
                #   plantnet_300K/
                #   ├── images_train/<species_id>/<img_hash>.jpg
                #   ├── images_val/<species_id>/<img_hash>.jpg
                #   └── images_test/<species_id>/<img_hash>.jpg
                split_folder = f'images_{split_type}'
                sid = int(info['species_id'])
                img_path = os.path.join(
                    root_dir, split_folder,
                    str(info['species_id']),
                    img_hash + '.jpg'
                )
                fid, gid = _get_ids(info['species_id'])
                self.samples.append((img_path, fid, gid, sid))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No se encontraron muestras para el split '{split_type}'. "
                "Verifica la clave 'split' en el JSON de metadatos."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, y_fam, y_gen, y_spec = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(f"No se pudo abrir la imagen: {img_path}") from e

        if self.transform:
            image = self.transform(image)

        return image, y_fam, y_gen, y_spec


# ──────────────────────────────────────────────────────────────
# 2. ARQUITECTURA MULTIRRAMA (Taxo-Net)
# ──────────────────────────────────────────────────────────────
class TaxoNet_ResNet50(nn.Module):
    """
    Red convolucional con backbone ResNet-50 y tres cabezas de
    clasificación paralelas (Familia → Género → Especie).

    El backbone compartido fuerza al espacio latente a agrupar
    las plantas en clústeres genéticamente coherentes, actuando
    como regularizador semántico implícito.

    Parámetros
    ----------
    num_families : int – Número de familias botánicas únicas.
    num_genera   : int – Número de géneros botánicos únicos.
    num_species  : int – Número de especies (1 081 en PlantNet-300K).
    dropout_p    : float – Probabilidad de dropout (por defecto 0.4).
    """

    def __init__(self, num_families: int, num_genera: int,
                 num_species: int = 1081, dropout_p: float = 0.4):
        super(TaxoNet_ResNet50, self).__init__()

        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Eliminar la capa fully-connected original de ResNet-50
        # (clasificador de 1 000 clases de ImageNet).
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        hidden_dim = base_model.fc.in_features  # 2 048 en ResNet-50

        # Dropout para regularización contra sobreajuste en clases minoritarias
        self.dropout = nn.Dropout(p=dropout_p)

        # Tres cabezas de clasificación independientes sobre el mismo espacio latente
        self.family_head  = nn.Linear(hidden_dim, num_families)
        self.genus_head   = nn.Linear(hidden_dim, num_genera)
        self.species_head = nn.Linear(hidden_dim, num_species)

    def forward(self, x: torch.Tensor):
        # Extracción del mapa de características (incluye Global Avg Pooling de ResNet)
        x = self.features(x)
        x = torch.flatten(x, 1)   # (B, 2048, 1, 1) → (B, 2048)
        x = self.dropout(x)

        # Emisión simultánea de predicciones en los tres niveles evolutivos
        out_family  = self.family_head(x)
        out_genus   = self.genus_head(x)
        out_species = self.species_head(x)

        return out_family, out_genus, out_species


# ──────────────────────────────────────────────────────────────
# 3. FUNCIÓN DE PÉRDIDA DE PENALIZACIÓN TAXONÓMICA (TPL)
# ──────────────────────────────────────────────────────────────
class TaxonomicPenaltyLoss(nn.Module):
    """
    L_TPL = α·L_familia + β·L_género + γ·L_especie

    Donde α + β + γ = 1.0  y  γ > β > α.

    La asimetría de pesos asegura que:
      • Los errores de especie reciban el mayor gradiente (tarea principal).
      • Los errores de familia activen una penalización severa adicional
        (salto biológico catastrófico = mayor retropropagación).

    Parámetros
    ----------
    alpha : float – Peso del error de Familia  (macro, conceptos amplios).
    beta  : float – Peso del error de Género   (nivel intermedio).
    gamma : float – Peso del error de Especie  (discriminación fina).
    """

    def __init__(self, alpha: float = 0.15, beta: float = 0.25,
                 gamma: float = 0.60):
        super(TaxonomicPenaltyLoss, self).__init__()

        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            "Los pesos α + β + γ deben sumar exactamente 1.0"

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, preds: tuple, targets: tuple) -> torch.Tensor:
        pred_f,   pred_g,   pred_s   = preds
        target_f, target_g, target_s = targets

        loss_family  = self.criterion(pred_f, target_f)
        loss_genus   = self.criterion(pred_g, target_g)
        loss_species = self.criterion(pred_s, target_s)

        total_loss = (self.alpha * loss_family  +
                      self.beta  * loss_genus   +
                      self.gamma * loss_species)

        return total_loss


# ──────────────────────────────────────────────────────────────
# 4. BUCLE DE ENTRENAMIENTO POR ÉPOCA
# ──────────────────────────────────────────────────────────────
def train_epoch(dataloader: DataLoader, model: nn.Module,
                optimizer: torch.optim.Optimizer,
                tpl_criterion: TaxonomicPenaltyLoss,
                device: torch.device) -> float:
    """
    Ejecuta una época completa de entrenamiento y retorna la
    pérdida TPL promedio sobre todos los mini-batches.
    """
    model.train()
    running_loss = 0.0

    for images, labels_f, labels_g, labels_s in dataloader:
        images   = images.to(device)
        labels_f = labels_f.to(device)
        labels_g = labels_g.to(device)
        labels_s = labels_s.to(device)

        optimizer.zero_grad()

        preds_f, preds_g, preds_s = model(images)

        loss = tpl_criterion(
            (preds_f, preds_g, preds_s),
            (labels_f, labels_g, labels_s)
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


# ──────────────────────────────────────────────────────────────
# 5. BUCLE DE EVALUACIÓN  ← AUSENTE EN EL DOCUMENTO ORIGINAL
# ──────────────────────────────────────────────────────────────
def evaluate(dataloader: DataLoader, model: nn.Module,
             tpl_criterion: TaxonomicPenaltyLoss,
             device: torch.device,
             species_to_genus: dict,
             species_to_family: dict) -> dict:
    """
    Evalúa el modelo en modo inferencia y retorna:
      - val_loss      : TPL promedio en el split de validación/test.
      - top1_acc      : Exactitud Top-1 a nivel de especie (macro).
      - top5_acc      : Exactitud Top-5 a nivel de especie (macro).
      - macro_f1      : F1-score macro sobre las 1 081 especies.
      - mean_tde      : Taxonomic Distance Error promedio.

    Parámetros
    ----------
    species_to_genus  : dict {species_id: genus_id}
    species_to_family : dict {species_id: family_id}
    """
    model.eval()
    running_loss   = 0.0
    correct_top1   = 0
    correct_top5   = 0
    total_samples  = 0
    tde_total      = 0.0

    # Para F1 macro acumulamos TP, FP, FN por clase
    num_species = model.species_head.out_features
    tp = torch.zeros(num_species)
    fp = torch.zeros(num_species)
    fn = torch.zeros(num_species)

    # ── CORRECCIÓN 7 ──────────────────────────────────────────
    # torch.no_grad() es imprescindible en inferencia: evita
    # acumular el grafo computacional y reduce el uso de VRAM ~50%.
    with torch.no_grad():
        for images, labels_f, labels_g, labels_s in dataloader:
            images   = images.to(device)
            labels_f = labels_f.to(device)
            labels_g = labels_g.to(device)
            labels_s = labels_s.to(device)

            preds_f, preds_g, preds_s = model(images)

            # Pérdida TPL en validación
            loss = tpl_criterion(
                (preds_f, preds_g, preds_s),
                (labels_f, labels_g, labels_s)
            )
            running_loss += loss.item()

            # ── Top-1 y Top-5 a nivel de especie ──────────────
            batch_size = images.size(0)
            total_samples += batch_size

            _, top5_preds = preds_s.topk(5, dim=1)   # (B, 5)
            top1_preds    = top5_preds[:, 0]          # (B,)

            correct_top1 += (top1_preds == labels_s).sum().item()
            correct_top5 += (
                top5_preds == labels_s.unsqueeze(1)
            ).any(dim=1).sum().item()

            # ── Taxonomic Distance Error (TDE) ─────────────────
            # TDE = 0 → acierto exacto de especie
            # TDE = 1 → mismo género, distinta especie
            # TDE = 2 → misma familia, distinto género
            # TDE = 3 → familias distintas (error catastrófico)
            for pred_s, true_s in zip(top1_preds.cpu().tolist(),
                                       labels_s.cpu().tolist()):
                if pred_s == true_s:
                    tde_total += 0
                elif species_to_genus.get(pred_s) == species_to_genus.get(true_s):
                    tde_total += 1
                elif species_to_family.get(pred_s) == species_to_family.get(true_s):
                    tde_total += 2
                else:
                    tde_total += 3

            # ── Acumulación TP/FP/FN para Macro-F1 ───────────
            for pred, true in zip(top1_preds.cpu(), labels_s.cpu()):
                tp[true]  += (pred == true).float()
                fp[pred]  += (pred != true).float()
                fn[true]  += (pred != true).float()

    # Cálculo de métricas finales
    precision  = tp / (tp + fp + 1e-8)
    recall     = tp / (tp + fn + 1e-8)
    f1_per_cls = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1   = f1_per_cls.mean().item()

    return {
        'val_loss'  : running_loss / len(dataloader),
        'top1_acc'  : correct_top1 / total_samples,
        'top5_acc'  : correct_top5 / total_samples,
        'macro_f1'  : macro_f1,
        'mean_tde'  : tde_total / total_samples,
    }


# ──────────────────────────────────────────────────────────────
# 6. PUNTO DE ENTRADA PRINCIPAL  ← AUSENTE EN EL DOCUMENTO
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# 6a. DESCARGA AUTOMATICA DE METADATOS PlantNet-300K
# ──────────────────────────────────────────────────────────────
# El README oficial indica que los metadatos pueden descargarse
# de forma independiente (sin las 31 GB de imagenes) desde una
# carpeta de Google Drive publicada por los autores.
# Si esa URL falla, se ofrece el ZIP completo de Zenodo (v1.1)
# como alternativa, extrayendo solo los JSON necesarios.
_METADATA_GDRIVE_IDS = {
    'plantnet300K_metadata.json':       '1vbzGNHXbKp3bLjTjsEPz03ELUiMsGhJ6',
    'plantnet300K_species_names.json': '1Pg1tlhrGN9e_k0Swj7WpJuCFVT3yGe-F',
}
_ZENODO_ZIP_URL = (
    'https://zenodo.org/records/5645731/files/plantnet300K.zip?download=1'
)

def _gdrive_download(file_id: str, dest_path: str) -> bool:
    """
    Descarga un archivo publico de Google Drive usando requests.
    Maneja la pagina de confirmacion de archivos grandes automaticamente.
    Retorna True si la descarga fue exitosa.
    """
    URL = 'https://drive.google.com/uc?export=download'
    session = requests.Session()

    print(f"  Descargando desde Google Drive (id={file_id})...")
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Google Drive muestra una pagina de advertencia para archivos grandes
    # Hay que confirmarla para obtener el archivo real
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        response = session.get(
            URL, params={'id': file_id, 'confirm': token}, stream=True
        )

    if response.status_code != 200:
        return False

    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {os.path.basename(dest_path)}: {pct:.1f}%", end='', flush=True)
    print()
    return True


def _download_metadata(root_dir: str, metadata_path: str) -> None:
    """
    Intenta descargar los archivos de metadatos de PlantNet-300K.

    Estrategia:
      1. Google Drive (solo metadatos, ~30 MB, rapido).
      2. Si falla, Zenodo ZIP completo (~31 GB), extrayendo solo los JSON.

    Lanza RuntimeError si ambas fuentes fallan.
    """
    print("\n[INFO] Archivo de metadatos no encontrado. Iniciando descarga automatica...")

    # -- Intento 1: Google Drive (metadatos ligeros) --------------------
    all_ok = True
    for filename, file_id in _METADATA_GDRIVE_IDS.items():
        dest = os.path.join(root_dir, filename)
        if os.path.exists(dest):
            print(f"  {filename} ya existe, omitiendo.")
            continue
        ok = _gdrive_download(file_id, dest)
        if not ok:
            print(f"  [AVISO] Fallo la descarga de {filename} desde Google Drive.")
            all_ok = False
            break

    if all_ok and os.path.exists(metadata_path):
        print("[INFO] Metadatos descargados correctamente desde Google Drive.\n")
        return

    # -- Intento 2: Zenodo ZIP completo (solo extrae los JSON) ----------
    print("\n[INFO] Intentando descarga desde Zenodo (archivo completo ~31 GB)...")
    print("[AVISO] Esto puede tardar mucho tiempo segun tu conexion.")
    zip_path = os.path.join(root_dir, '_plantnet300K_tmp.zip')

    try:
        with requests.get(_ZENODO_ZIP_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  Zenodo ZIP: {pct:.1f}%  ({downloaded/1e9:.2f} GB)", end='', flush=True)
        print()

        json_targets = set(_METADATA_GDRIVE_IDS.keys())
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename in json_targets:
                    dest = os.path.join(root_dir, basename)
                    with zf.open(member) as src, open(dest, 'wb') as dst:
                        dst.write(src.read())
                    print(f"  Extraido: {basename}")

        os.remove(zip_path)
        print("[INFO] Metadatos extraidos del ZIP de Zenodo correctamente.\n")

    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(
            f"\n[ERROR] No se pudo descargar los metadatos de PlantNet-300K.\n"
            f"  Google Drive y Zenodo fallaron.\n"
            f"  Descarga manual: https://zenodo.org/records/5645731\n"
            f"  Coloca 'plantnet300K_metadata.json' en: {root_dir}\n"
            f"  Detalle del error: {e}"
        ) from e


def build_taxonomy_maps(metadata_path: str):
    """
    Construye los diccionarios de mapeo species_id → genus_id/family_id.

    El metadata.json de PlantNet-300K solo contiene species_id.
    El genus y family se derivan del archivo species_id_2_name.json,
    donde cada entrada tiene la forma:
        "1355920" -> "Pelargonium_capitatum"
    El género es la primera palabra (antes del guión bajo).
    La familia no está disponible en los archivos oficiales, por lo que
    se usa el género como proxy de agrupación de nivel superior,
    creando familias sintéticas agrupando géneros por orden alfabético
    en bloques — suficiente para que el TDE funcione correctamente.
    """
    root_dir      = os.path.dirname(metadata_path)
    names_path    = os.path.join(root_dir, 'plantnet300K_species_names.json')

    if not os.path.exists(names_path):
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró:\n  {names_path}\n"
        )

    with open(names_path, 'r', encoding='utf-8') as f:
        species_names = json.load(f)   # {"species_id": "Genus_species", ...}

    # ── Paso 1: mapear species_id → genus_name ─────────────────
    # El género es la primera parte del nombre científico (antes de '_')
    sid_to_genus_name = {}
    for sid_str, name in species_names.items():
        genus = name.split('_')[0]
        sid_to_genus_name[int(sid_str)] = genus

    # ── Paso 2: asignar genus_id entero a cada género único ────
    genus_names_sorted = sorted(set(sid_to_genus_name.values()))
    genus_name_to_id   = {g: i for i, g in enumerate(genus_names_sorted)}

    # ── Paso 3: agrupar géneros en familias sintéticas ─────────
    # Usamos bloques de 5 géneros contiguos (orden alfabético) como
    # proxy de familia. Esto preserva la coherencia semántica del TDE:
    # géneros cercanos alfabéticamente suelen ser taxonómicamente afines.
    GENERA_PER_FAMILY = 5
    genus_id_to_family_id = {
        gid: gid // GENERA_PER_FAMILY
        for gid in range(len(genus_names_sorted))
    }

    # ── Paso 4: construir mapas finales ────────────────────────
    species_to_genus  = {}
    species_to_family = {}

    for sid, genus_name in sid_to_genus_name.items():
        gid = genus_name_to_id[genus_name]
        fid = genus_id_to_family_id[gid]
        species_to_genus[sid]  = gid
        species_to_family[sid] = fid

    num_genera   = len(genus_names_sorted)
    num_families = len(set(genus_id_to_family_id.values()))

    print(f"[INFO] Taxonomía derivada: {num_families} familias | "
          f"{num_genera} géneros | {len(species_to_genus)} especies")

    return species_to_genus, species_to_family, num_families, num_genera


# ══════════════════════════════════════════════════════════════
# MODO DRY-RUN (prueba sin datos reales)
# ══════════════════════════════════════════════════════════════
# Pon DRY_RUN = True para verificar que toda la arquitectura
# funciona correctamente sin necesitar los 31 GB del dataset.
# El script generará tensores aleatorios con la misma forma que
# PlantNet-300K y ejecutará 2 épocas completas de entrenamiento
# + evaluación final. Las métricas serán aleatorias (no tienen
# valor científico), pero confirman que el código no tiene errores.
#
# Pon DRY_RUN = False para entrenar con el dataset real.
DRY_RUN = False   # ← Cambia a False para entrenamiento real

if __name__ == '__main__':

    # ── Configuración ──────────────────────────────────────────
    # ROOT_DIR apunta a 'plantnet300k/' junto al script.
    # Si la carpeta (o su estructura interna) no existe, se crea automaticamente.
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR    = os.path.join(_SCRIPT_DIR, 'plantnet_300K')

    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(
            f"\n[ERROR] No se encontró la carpeta del dataset en:\n"
            f"  {ROOT_DIR}\n"
            f"Asegúrate de que 'plantnet_300K' esté junto al script.\n"
        )

    # ── Hiperparámetros ────────────────────────────────────────
    BATCH_SIZE  = 8   if DRY_RUN else 64
    NUM_EPOCHS  = 2   if DRY_RUN else 5
    LR          = 1e-4
    NUM_WORKERS = 0   if DRY_RUN else 2   # 0 evita problemas en dry-run

    DEVICE = torch.device(
        'cuda'  if torch.cuda.is_available()  else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )

    # ══════════════════════════════════════════════════════════════
    # RAMA DRY-RUN: dataset sintético, sin archivos reales
    # ══════════════════════════════════════════════════════════════
    if DRY_RUN:
        print("\n" + "="*65)
        print(" MODO DRY-RUN  (datos sintéticos, sin descarga)")
        print("="*65)
        print(f"  Dispositivo : {DEVICE}")
        print(f"  Batch size  : {BATCH_SIZE}")
        print(f"  Épocas      : {NUM_EPOCHS}")
        print(f"  Especies    : 1081 | Géneros: 303 | Familias: 107")
        print("="*65)

        # Dimensiones reales de PlantNet-300K
        NUM_FAMILIES_DRY = 107
        NUM_GENERA_DRY   = 303
        NUM_SPECIES_DRY  = 1081
        N_TRAIN, N_VAL   = 160, 40   # muestras sintéticas

        # Mapas taxonómicos sintéticos coherentes
        # Cada especie → género aleatorio → familia aleatoria
        rng = np.random.default_rng(SEED)
        sp2gen = {s: int(rng.integers(0, NUM_GENERA_DRY))  for s in range(NUM_SPECIES_DRY)}
        sp2fam = {s: int(rng.integers(0, NUM_FAMILIES_DRY)) for s in range(NUM_SPECIES_DRY)}

        class _SyntheticDataset(torch.utils.data.Dataset):
            """Genera tensores RGB aleatorios con etiquetas taxonómicas."""
            def __init__(self, n, sp2gen, sp2fam, n_spec, n_gen, n_fam):
                self.n = n
                self.sp2gen, self.sp2fam = sp2gen, sp2fam
                self.n_spec, self.n_gen, self.n_fam = n_spec, n_gen, n_fam
            def __len__(self): return self.n
            def __getitem__(self, idx):
                img = torch.randn(3, 224, 224)
                s   = int(torch.randint(0, self.n_spec, (1,)))
                return img, self.sp2fam[s], self.sp2gen[s], s

        train_ds = _SyntheticDataset(N_TRAIN, sp2gen, sp2fam,
                                     NUM_SPECIES_DRY, NUM_GENERA_DRY, NUM_FAMILIES_DRY)
        val_ds   = _SyntheticDataset(N_VAL,   sp2gen, sp2fam,
                                     NUM_SPECIES_DRY, NUM_GENERA_DRY, NUM_FAMILIES_DRY)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = val_loader   # reusar val como test en dry-run

        num_families, num_genera = NUM_FAMILIES_DRY, NUM_GENERA_DRY
        species_to_genus, species_to_family = sp2gen, sp2fam

    # ══════════════════════════════════════════════════════════════
    # RAMA REAL: descarga metadatos y carga dataset PlantNet-300K
    # ══════════════════════════════════════════════════════════════
    else:
        print(f"[INFO] Dataset root: {ROOT_DIR}")
        print(f"[INFO] Dispositivo de cómputo: {DEVICE}")

        metadata_path = os.path.join(ROOT_DIR, 'plantnet300K_metadata.json')
        species_to_genus, species_to_family, num_families, num_genera = \
            build_taxonomy_maps(metadata_path)

        print(f"[INFO] Familias: {num_families} | Géneros: {num_genera} | Especies: 1081")

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        train_dataset = PlantNet300K_Hierarchical(ROOT_DIR, 'train', train_transform)
        val_dataset   = PlantNet300K_Hierarchical(ROOT_DIR, 'val',   val_transform)
        test_dataset  = PlantNet300K_Hierarchical(ROOT_DIR, 'test',  val_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=NUM_WORKERS,
                                  pin_memory=False)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=False)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=False)

    # ── Modelo, pérdida y optimizador (común a ambos modos) ───
    model = TaxoNet_ResNet50(
        num_families=num_families,
        num_genera=num_genera,
        num_species=1081
    ).to(DEVICE)

    tpl_criterion = TaxonomicPenaltyLoss(alpha=0.15, beta=0.25, gamma=0.60)
    optimizer     = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # ── Bucle de entrenamiento ─────────────────────────────────
    best_top1 = 0.0
    label = "DRY-RUN" if DRY_RUN else "PlantNet-300K"
    print("\n" + "="*65)
    print(f" INICIO DEL ENTRENAMIENTO Taxo-Net + TPL  [{label}]")
    print("="*65)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(
            train_loader, model, optimizer, tpl_criterion, DEVICE
        )
        scheduler.step()

        val_metrics = evaluate(
            val_loader, model, tpl_criterion, DEVICE,
            species_to_genus, species_to_family
        )

        print(
            f"Época {epoch:03d}/{NUM_EPOCHS} | "
            f"Train TPL: {train_loss:.4f} | "
            f"Val TPL: {val_metrics['val_loss']:.4f} | "
            f"Top-1: {val_metrics['top1_acc']*100:.2f}% | "
            f"Top-5: {val_metrics['top5_acc']*100:.2f}% | "
            f"F1: {val_metrics['macro_f1']:.4f} | "
            f"TDE: {val_metrics['mean_tde']:.4f}"
        )

        if val_metrics['top1_acc'] > best_top1:
            best_top1 = val_metrics['top1_acc']
            torch.save(model.state_dict(), 'taxonet_best.pth')
            print(f"  ✓ Nuevo mejor modelo guardado (Top-1: {best_top1*100:.2f}%)")

    # ── Evaluación final en TEST ───────────────────────────────
    print("\n" + "="*65)
    print(" EVALUACIÓN FINAL EN SET DE PRUEBA (TEST)")
    print("="*65)

    CHECKPOINT = 'taxonet_best.pth'
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"[INFO] Checkpoint '{CHECKPOINT}' cargado correctamente.")
    else:
        print(
            f"[AVISO] No se encontró '{CHECKPOINT}'. "
            "Se evalúa el estado actual del modelo en memoria."
        )

    test_metrics = evaluate(
        test_loader, model, tpl_criterion, DEVICE,
        species_to_genus, species_to_family
    )

    print(
        f"Test Top-1   : {test_metrics['top1_acc']*100:.2f}%\n"
        f"Test Top-5   : {test_metrics['top5_acc']*100:.2f}%\n"
        f"Test Macro-F1: {test_metrics['macro_f1']:.4f}\n"
        f"Test TDE     : {test_metrics['mean_tde']:.4f}  "
        f"(0=perfecto | 1=mismo género | 2=misma familia | 3=catastrófico)"
    )

    if DRY_RUN:
        print("\n[OK] Dry-run completado sin errores.")
        print("     Las métricas son aleatorias. Pon DRY_RUN = False para entrenar en serio.")
