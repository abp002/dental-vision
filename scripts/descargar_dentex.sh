#!/bin/sh
# Descarga el dataset DENTEX 2023 (Zenodo, CC BY 4.0) y extrae solo los
# subconjuntos etiquetados que usa el proyecto.
#
# Son ~10 GB comprimidos y unos 5 GB extraidos. El subconjunto 'unlabelled'
# (1.571 imagenes para preentrenamiento) no se extrae: no lo usamos todavia y
# duplicaria el espacio en disco.
#
# Uso:  sh scripts/descargar_dentex.sh
set -e

DESTINO="$(cd "$(dirname "$0")/.." && pwd)/data/dentex"
ZIP="$DESTINO/training_data.zip"
URL="https://zenodo.org/records/7812323/files/training_data.zip?download=1"

mkdir -p "$DESTINO"

if [ -f "$ZIP" ]; then
    echo "Ya existe $ZIP, no se vuelve a descargar."
else
    echo "Descargando DENTEX (~10 GB)..."
    curl -L -# -o "$ZIP" "$URL"
fi

echo "Verificando integridad..."
unzip -t "$ZIP" > /dev/null

echo "Extrayendo subconjuntos etiquetados..."
unzip -q -o "$ZIP" \
    "training_data/quadrant_enumeration/*" \
    "training_data/quadrant-enumeration-disease/*" \
    -d "$DESTINO"

echo "Listo. Radiografias en $DESTINO/training_data/"
find "$DESTINO/training_data" -name "*.png" | wc -l | xargs echo "PNG disponibles:"
