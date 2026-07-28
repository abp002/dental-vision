"""Entrena el detector de piezas dentales.

Uso:
    uv run python scripts/entrenar.py --epocas 12 --ancho 800 --lote 2

Guarda un punto de control por época en modelos/ y registra las métricas de
validación en modelos/historial.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import torch
from torch.utils.data import DataLoader

# El benchmark mostro que la CPU es 17x mas rapida que MPS con Mask R-CNN:
# sus operaciones dispersas (roi_align, nms) no tienen kernel Metal y hacen
# fallback a CPU con copias constantes. Se fija el numero de hilos para
# aprovechar los nucleos de rendimiento sin saturar la maquina.
torch.set_num_threads(10)

from dentalvision.datos import DientesDataset, colacion
from dentalvision.dentex import Subconjunto
from dentalvision.evaluacion import evaluar
from dentalvision.modelo import crear, dispositivo
from dentalvision.particiones import particionar


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--epocas", type=int, default=12)
    p.add_argument("--ancho", type=int, default=800)
    p.add_argument("--lote", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--limite", type=int, default=0, help="usar solo N imagenes (pruebas)")
    p.add_argument("--dispositivo", default="cpu",
                   help="cpu por defecto: MPS es 17x mas lento con Mask R-CNN")
    return p.parse_args()


@torch.no_grad()
def validar(modelo, cargador, dev):
    modelo.eval()
    preds, reales = [], []
    for imgs, tgts in cargador:
        salida = modelo([x.to(dev) for x in imgs])
        preds.extend({k: v.detach() for k, v in s.items()} for s in salida)
        reales.extend(tgts)
    return evaluar(preds, reales)


def main() -> int:
    a = argumentos()
    dev = torch.device(a.dispositivo) if a.dispositivo else dispositivo()

    sub = Subconjunto("enumeracion")
    part = particionar(sub)
    ent, val = part.entrenamiento, part.validacion
    if a.limite:
        ent, val = ent[: a.limite], val[: max(2, a.limite // 4)]

    print(f"dispositivo: {dev}   entrenamiento: {len(ent)}   validacion: {len(val)}")

    ds_ent = DientesDataset(sub, ent, ancho=a.ancho, aumentar=True)
    ds_val = DientesDataset(sub, val, ancho=a.ancho, aumentar=False)
    dl_ent = DataLoader(ds_ent, batch_size=a.lote, shuffle=True, collate_fn=colacion)
    dl_val = DataLoader(ds_val, batch_size=a.lote, shuffle=False, collate_fn=colacion)

    modelo = crear().to(dev)
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas)

    destino = RAIZ / "modelos"
    destino.mkdir(exist_ok=True)
    historial = destino / "historial.jsonl"
    mejor = -1.0

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        t0, suma, n = time.perf_counter(), 0.0, 0
        for i, (imgs, tgts) in enumerate(dl_ent):
            imgs = [x.to(dev) for x in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            perdidas = modelo(imgs, tgts)
            total = sum(perdidas.values())
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            suma += total.item()
            n += 1
            if i % 20 == 0:
                print(f"  epoca {epoca} lote {i}/{len(dl_ent)} perdida {total.item():.3f}", flush=True)
        plan.step()

        res = validar(modelo, dl_val, dev)
        dt = time.perf_counter() - t0
        print(f"epoca {epoca}/{a.epocas}  perdida {suma/max(1,n):.3f}  {res}  [{dt/60:.1f} min]", flush=True)

        with historial.open("a") as f:
            f.write(json.dumps({
                "epoca": epoca, "perdida": suma / max(1, n),
                "cobertura": res.cobertura, "numeracion": res.numeracion,
                "perfectas": res.perfectas, "minutos": dt / 60,
            }) + "\n")

        torch.save(modelo.state_dict(), destino / "ultimo.pt")
        if res.numeracion > mejor:
            mejor = res.numeracion
            torch.save(modelo.state_dict(), destino / "mejor.pt")
            print(f"  nuevo mejor: numeracion {mejor:.1%}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
