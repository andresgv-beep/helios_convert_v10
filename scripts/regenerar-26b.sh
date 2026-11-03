#!/usr/bin/env bash
# Regenera el HNF canónico de Gemma 4 26B MoE (texto HQ8K + expertos HQ44 refinados con GPTQ
# en Héctor + visión FP16) desde el checkpoint original, y lo compara con el canónico.
#
# uso: scripts/regenerar-26b.sh checkpoint/ entradas/ trabajo/ salida.v10.hnf [canonico.v10.hnf]
#
#   checkpoint/  gemma-4-26b-a4b-it de Hugging Face (config, processor_config, safetensors,
#                tokenizer.json, chat_template.jinja)
#   entradas/    calibracion/ (los 181 casos .ids), importancia-hector-v1/ (layerN.gate_up.f32 y
#                layerN.down.f32) y SHA256SUMS. Copia en el NAS: compartido/modelo-26b/regeneracion/
#   trabajo/     carpeta para los intermedios (pico de unos 34 GB; se borran al terminar cada paso)
#
# Necesita compilados `hnf10` (cargo build --release) y `hnf_refine` (Hector-engine/build). El
# refinado corre en la 4070 Ti (bus PCI 1) y tarda unos 41 minutos; el resto, unos 20.
# Docs: docs/HNFV10.md §13 (cadena bit a bit) y §15 (visión).
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
    sed -n '5,13p' "$0" >&2
    exit 2
fi
checkpoint=$(realpath -- "$1")
entradas=$(realpath -- "$2")
trabajo=$(realpath -m -- "$3")
salida=$(realpath -m -- "$4")
canonico=${5:+$(realpath -- "$5")}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
hnf10=$repo/target/release/hnf10
refine=${HNF_REFINE:-$repo/../Hector-engine/build/hnf_refine}
commit=$(git -C "$repo" rev-parse --short HEAD)

[[ -x $hnf10 && -x $refine ]] || { echo "faltan $hnf10 o $refine compilados" >&2; exit 2; }
[[ -e $salida ]] && { echo "$salida ya existe" >&2; exit 2; }
paso() { echo "== $(date +%T) $*"; }

paso "comprobar las entradas"
(cd "$entradas" && sha256sum --quiet -c SHA256SUMS)

mkdir -p "$trabajo"
texto=$trabajo/texto.v10.hnf
refinado=$trabajo/refinado.v10.hnf

paso "convertir con la importancia de Héctor (13 min)"
[[ -e $texto ]] || "$hnf10" convert --model "$checkpoint" --out "$texto" --dense-hq8 \
    --importance "$entradas/importancia-hector-v1" --commit "$commit"

paso "refinar los expertos con GPTQ en la GPU (41 min)"
if [[ ! -e $refinado ]]; then
    "$hnf10" begin-edit --from "$texto" --to "$refinado"
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${REFINE_GPU:-1} \
    "$refine" "$texto" "$refinado" --model "$checkpoint" --calibration "$entradas/calibracion" \
    --damp 1 --work "$trabajo/refine"
"$hnf10" seal "$refinado" --record "$trabajo/refine/record.json"
rm -f -- "$texto"

paso "añadir la visión (2 min)"
"$hnf10" add-vision --from "$refinado" --model "$checkpoint" --to "$salida" --commit "$commit"
rm -f -- "$refinado"
(cd "$(dirname "$salida")" && sha256sum "$(basename "$salida")" > "$(basename "$salida").sha256")

if [[ -n $canonico ]]; then
    paso "comparar con el canónico"
    "$hnf10" compare "$canonico" "$salida"
fi
paso "listo: $salida"
