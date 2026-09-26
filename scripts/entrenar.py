"""Entrena el detector de piezas dentales.

Uso:
    uv run python scripts/entrenar.py --nombre base

Cada ejecución guarda en modelos/<nombre>/ su configuración, el historial de
validación por época (historial.jsonl), el último punto de control y el mejor
según errores por radiografía en validación.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dentalvision.datos import ANCHO, DientesDataset, colacion
from dentalvision.dentex import Subconjunto
from dentalvision.evaluacion import evaluar
from dentalvision.modelo import crear, dispositivo, sin_mascaras
from dentalvision.particiones import particionar

RAIZ = Path(__file__).resolve().parents[1]


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--nombre", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--epocas", type=int, default=24)
    p.add_argument("--ancho", type=int, default=ANCHO)
    p.add_argument("--lote", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--calentamiento", type=int, default=300,
                   help="pasos de subida lineal del lr al principio")
    p.add_argument("--rois", type=int, default=128, help="RoIs muestreadas por imagen")
    p.add_argument("--congelar-bn", action="store_true",
                   help="BatchNorm del backbone fija (FrozenBatchNorm2d)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--trabajadores", type=int, default=6, help="procesos del DataLoader")
    p.add_argument("--limite", type=int, default=0, help="usar solo N imagenes (pruebas)")
    p.add_argument("--dispositivo", default=None,
                   help="cuda, cpu o mps (por defecto cuda si hay GPU NVIDIA)")
    p.add_argument("--sin-amp", action="store_true",
                   help="desactiva la precision mixta en CUDA")
    return p.parse_args()


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


@torch.no_grad()
def validar(modelo, cargador, dev):
    modelo.eval()
    preds, reales = [], []
    with sin_mascaras(modelo):
        for imgs, tgts in cargador:
            salida = modelo([x.to(dev, non_blocking=True) for x in imgs])
            # A CPU en cuanto salen: acumularlas en la GPU la llena sin necesidad.
            preds.extend({k: v.cpu() for k, v in s.items()} for s in salida)
            reales.extend(tgts)
    return evaluar(preds, reales)


def main() -> int:
    a = argumentos()
    dev = torch.device(a.dispositivo) if a.dispositivo else dispositivo()
    amp = dev.type == "cuda" and not a.sin_amp
    fijar_semilla(a.semilla)

    sub = Subconjunto("enumeracion")
    part = particionar(sub)
    ent, val = part.entrenamiento, part.validacion
    if a.limite:
        ent, val = ent[: a.limite], val[: max(2, a.limite // 4)]

    destino = RAIZ / "modelos" / a.nombre
    destino.mkdir(parents=True, exist_ok=False)  # no pisar ejecuciones anteriores
    (destino / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n")
    historial = destino / "historial.jsonl"

    print(f"dispositivo: {dev}  amp: {amp}  entrenamiento: {len(ent)}  "
          f"validacion: {len(val)}  -> {destino}", flush=True)

    ds_ent = DientesDataset(sub, ent, ancho=a.ancho, aumentar=True)
    ds_val = DientesDataset(sub, val, ancho=a.ancho, con_mascaras=False)
    comunes = dict(
        batch_size=a.lote,
        collate_fn=colacion,
        num_workers=a.trabajadores,
        persistent_workers=a.trabajadores > 0,
        pin_memory=dev.type == "cuda",
    )
    dl_ent = DataLoader(ds_ent, shuffle=True, **comunes)
    dl_val = DataLoader(ds_val, shuffle=False, **comunes)

    modelo = crear(ancho=a.ancho, rois=a.rois, congelar_bn=a.congelar_bn).to(dev)
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)

    # Subida lineal y después coseno, paso a paso. Las cabezas nuevas empiezan
    # con pesos aleatorios: un lr alto de golpe desordena el backbone preentrenado.
    total_pasos = a.epocas * len(dl_ent)

    def factor(paso: int) -> float:
        if paso < a.calentamiento:
            return (paso + 1) / a.calentamiento
        t = (paso - a.calentamiento) / max(1, total_pasos - a.calentamiento)
        return 0.5 * (1 + math.cos(math.pi * t))

    plan = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    escalador = torch.amp.GradScaler("cuda", enabled=amp)
    mejor = math.inf

    for epoca in range(1, a.epocas + 1):
        modelo.train()
        t0, suma, n = time.perf_counter(), 0.0, 0
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i, (imgs, tgts) in enumerate(dl_ent):
            imgs = [x.to(dev, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast(dev.type, enabled=amp):
                perdidas = modelo(imgs, tgts)
                total = sum(perdidas.values())
            if not math.isfinite(total.item()):
                print(f"perdida no finita en epoca {epoca} lote {i}: {perdidas}", flush=True)
                return 1
            opt.zero_grad(set_to_none=True)
            escalador.scale(total).backward()
            escalador.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            escalador.step(opt)
            escalador.update()
            plan.step()
            suma += total.item()
            n += 1
            if i % 50 == 0:
                print(f"  epoca {epoca} lote {i}/{len(dl_ent)} perdida {total.item():.3f}", flush=True)
        t_ent = time.perf_counter() - t0

        res = validar(modelo, dl_val, dev)
        dt = time.perf_counter() - t0
        memoria = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        print(f"epoca {epoca}/{a.epocas}  perdida {suma / max(1, n):.3f}  {res}  "
              f"[{dt / 60:.1f} min, {t_ent / max(1, n):.2f} s/paso, {memoria:.1f} GB]", flush=True)

        with historial.open("a") as f:
            f.write(json.dumps({
                "epoca": epoca,
                "perdida": suma / max(1, n),
                **res.metricas(),
                "minutos": dt / 60,
                "segundos_por_paso": t_ent / max(1, n),
                "memoria_gb": memoria,
            }) + "\n")

        torch.save(modelo.state_dict(), destino / "ultimo.pt")
        if res.errores_por_radiografia < mejor:
            mejor = res.errores_por_radiografia
            torch.save(modelo.state_dict(), destino / "mejor.pt")
            print(f"  nuevo mejor: {mejor:.2f} errores por radiografia", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
