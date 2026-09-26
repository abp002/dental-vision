#!/bin/sh
# Descarga el dataset DENTEX 2023 (Zenodo, CC BY 4.0) y extrae solo los
# subconjuntos etiquetados que usa el proyecto.
#
# Son ~10 GB comprimidos y unos 5 GB extraidos. El subconjunto 'unlabelled'
# (1.571 imagenes para preentrenamiento) no se extrae: no lo usamos todavia y
# duplicaria el espacio en disco.
#
# Zenodo puede ir muy lento (medido: ~0,6 MB/s). Los autores publican el mismo
# fichero en Hugging Face (ibrahimhamamci/DENTEX), bastante mas rapido. Se puede
# usar con DENTEX_URL; el MD5 de Zenodo garantiza que es identico byte a byte.
# Ojo: la ficha de Hugging Face declara CC BY-NC-SA 4.0 y el registro de
# Zenodo, CC BY 4.0.
#
# Uso:  sh scripts/descargar_dentex.sh
#       DENTEX_URL=https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/training_data.zip \
#         sh scripts/descargar_dentex.sh
set -e

DESTINO="$(cd "$(dirname "$0")/.." && pwd)/data/dentex"
ZIP="$DESTINO/training_data.zip"
URL="${DENTEX_URL:-https://zenodo.org/records/7812323/files/training_data.zip?download=1}"
MD5="eba0c44fa4c6e8ec221db3009eccc1f9"  # el que publica Zenodo para training_data.zip

md5_de() {
    if command -v md5sum > /dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
    else md5 -q "$1"; fi
}

mkdir -p "$DESTINO"

if [ -f "$ZIP" ]; then
    echo "Ya existe $ZIP, no se vuelve a descargar."
else
    echo "Descargando DENTEX (~10 GB) desde $URL"
    curl -L -# -o "$ZIP" "$URL"
fi

echo "Verificando MD5..."
if [ "$(md5_de "$ZIP")" != "$MD5" ]; then
    echo "El MD5 no coincide con el de Zenodo: descarga incompleta o corrupta."
    echo "Borra $ZIP y vuelve a ejecutar el script."
    exit 1
fi

echo "Extrayendo subconjuntos etiquetados..."
unzip -q -o "$ZIP" \
    "training_data/quadrant_enumeration/*" \
    "training_data/quadrant-enumeration-disease/*" \
    -d "$DESTINO"

echo "Listo. Radiografias en $DESTINO/training_data/"
find "$DESTINO/training_data" -name "*.png" | wc -l | xargs echo "PNG disponibles:"
