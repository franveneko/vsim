"""The three storyboards: one long landscape cut and two vertical ones.

On-screen narration is in Spanish; the HUD keeps Minecraft's own vocabulary.
Every number quoted on screen is read from the run that actually happened - see
`stats()` - so the videos cannot drift from the results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minesim.world import MILESTONES, N_MILESTONES
from render import audio, ui
from render import build as build_mod
from render.build import (FPS, LANDSCAPE, PORTRAIT, card, chart_scene,
                          connectome_frame, hero_scene, population_scene,
                          scripted_scene, still_scene)
from render.video import Subtitles, VideoWriter

OUT = Path(__file__).resolve().parent.parent / "out"

TITLE_COL = ui.INK
ACC = ui.ACCENT

# The hand-written route's best time, measured by minesim.scripted.
SCRIPTED_REFERENCE = 457


def champion(genomes_path: Path, z) -> np.ndarray:
    """The genome the showcase selected, falling back to the training best."""
    path = Path(genomes_path).with_name("champion.npy")
    return np.load(path) if path.exists() else z["best_genome"]


def stats(run_dir: Path = OUT) -> dict:
    """Everything the narration is allowed to claim, straight from the artefacts."""
    meta = json.loads((run_dir / "evolution.json").read_text())
    show = json.loads((run_dir / "showcase.json").read_text())
    hist = meta["history"]
    best = show["best"]
    return {
        "neurons": meta["n_neurons"],
        "connections": meta["n_connections"],
        "genome": meta["genome_size"],
        "pop": meta["pop_size"],
        "generations": len(hist),
        "history": hist,
        "best": best,
        "attempts": show["agents"],
        "total_attempts": show["agents"] * max(len(show.get("all", [])), 1),
        "final_best_ms": max(h["best_progress"] for h in hist),
        "final_mean_ms": hist[-1]["mean_progress"],
        "won": bool(best["won"]),
        "ticks": int(best["ticks"]),
        "milestone": ("COMPLETE" if best["progress"] >= N_MILESTONES
                      else MILESTONES[min(best["progress"], N_MILESTONES - 1)]),
        "world_seed": int(best["world_seed"]),
        "agent": int(best["agent"]),
        "scripted_ticks": SCRIPTED_REFERENCE,
        "endgame": show.get("endgame"),
        "endgame_stage": show.get("endgame_stage", N_MILESTONES - 1),
        "endgame_won": show.get("endgame_worlds_won", 0),
        "endgame_worlds": show.get("endgame_worlds", 0),
    }


def _result_lines(s: dict, small: bool = False):
    """The result, stated exactly as it came out - including the part that did
    not work."""
    a, b = (54, 30) if not small else (60, 34)
    if s["won"]:
        head = [("EL DRAGÓN CAE, DE UNA SENTADA", a, ACC),
                (ui.format_time(s["ticks"]), int(a * 1.7), TITLE_COL)]
    else:
        head = [("DE UNA SENTADA, HASTA AQUÍ", a, ACC),
                (s["milestone"].replace("_", " "), int(a * 1.5), TITLE_COL),
                (f"{s['best']['progress']} de {N_MILESTONES} hitos  ·  "
                 f"{s['total_attempts']:,} intentos".replace(",", "."), b, ui.DIM)]
    tail = []
    eg = s.get("endgame")
    if eg and s["endgame_won"]:
        tail = [("", 18, TITLE_COL),
                ("PERO PUESTA EN EL END", a, ui.WARN),
                ("mata al dragón", int(a * 1.3), TITLE_COL),
                (f"en {s['endgame_won']} de {s['endgame_worlds']} mundos", b, ui.DIM)]
    return head + tail


# ------------------------------------------------------------------- YouTube
def youtube(genomes_path: Path, out: Path, s: dict) -> Path:
    fmt = LANDSCAPE
    z = np.load(genomes_path)
    best_genome = champion(genomes_path, z)
    snaps = z["snapshots"]
    snap_gens = z["snapshot_gens"]
    silent = out.with_name(out.stem + "_silent.mp4")
    w = VideoWriter(silent, fmt.size, FPS)
    build_mod.ACTIVE_SUBS = subs = Subtitles(FPS)

    # 1. cold open
    hero_scene(w, fmt, best_genome, s["world_seed"], s["agent"], 320, s["attempts"],
               brain_seed=0, stride=1, start_tick=0,
               title="UNA MOSCA JUGANDO A MINECRAFT",
               sub="MaleCNS v1.0  ·  conectoma real",
               captions=[(0, 150, "Esto es un cerebro de mosca. Está jugando a un Minecraft."),
                         (150, 320, "Ni una sola conexión de esta red está inventada: "
                                    "están medidas, sinapsis a sinapsis.")])

    # 2. title
    card(w, fmt, [("UN CEREBRO DE MOSCA", 72, TITLE_COL),
                  ("APRENDE A PASARSE MINECRAFT", 72, ACC)], 4.5,
         foot="MaleCNS v1.0 · Janelia FlyEM + Google Research · Cell, septiembre de 2026")

    # 3. the connectome
    still_scene(w, fmt, connectome_frame(fmt), [
        (6.0, "El 3 de septiembre de 2026 se publicó el conectoma completo del sistema "
              "nervioso central de una mosca macho: unas 166.000 neuronas y 125 millones "
              "de sinapsis."),
        (5.5, "Los datos son públicos. Este proyecto se los descarga y los usa tal cual, "
              "sin retocar el cableado."),
        (6.0, "Simular las 166.000 neuronas para cientos de moscas a la vez no cabe en "
              "cuatro núcleos, así que nos quedamos con el circuito que decide: ojos, "
              "cuerpo pedunculado, complejo central y neuronas descendentes."),
        (5.0, f"{s['neurons']:,} neuronas. {s['connections']:,} conexiones medidas. "
              f"El signo de cada una sale de su neurotransmisor."),
    ])

    # 3b. where learning lives
    card(w, fmt, [("DÓNDE APRENDE UNA MOSCA", 52, ui.DIM),
                  ("En el cuerpo pedunculado: 4.064 células de Kenyon", 38, TITLE_COL),
                  ("que disparan de forma dispersa, 97 neuronas de salida,", 38, TITLE_COL),
                  ("y 340 neuronas dopaminérgicas que dicen", 38, TITLE_COL),
                  ("cuáles de esas sinapsis hay que debilitar.", 38, TITLE_COL),
                  ("", 14, TITLE_COL),
                  ("Ese circuito está aquí tal y como se midió.", 38, ACC)], 9.0)

    # 4. the sandbox
    card(w, fmt, [("EL PROBLEMA", 56, ui.DIM),
                  ("Minecraft de verdad no corre aquí:", 44, TITLE_COL),
                  ("sin GPU, sin cuenta, sin escritorio.", 44, TITLE_COL),
                  ("", 20, TITLE_COL),
                  ("Así que se reimplementa lo que lo hace difícil.", 40, ACC)], 6.0)
    hero_scene(w, fmt, best_genome, 4100, 0, 260, 32, stride=1,
               title="MINESIM  ·  el sandbox", sub="19 hitos, la ruta Any% completa",
               captions=[(0, 130, "Minar, craftear, el árbol de herramientas, los mobs, "
                                  "los tres mundos."),
                         (130, 260, "Y la condición de victoria de verdad: matar al Ender "
                                    "Dragon.")])

    # 4b. what a route actually looks like, played by hand
    card(w, fmt, [("LA RUTA, A MANO", 56, ui.DIM),
                  ("Primero comprobamos que el mundo se puede pasar:", 38, TITLE_COL),
                  ("una política escrita a mano, sin aprender nada.", 38, TITLE_COL)], 5.0)
    scripted_scene(w, fmt, 4300, 700, stride=1,
                   title="RUTA DE REFERENCIA", sub="política escrita a mano, sin cerebro",
                   captions=[(0, 200, "Madera, banco, pico. Siempre en el mismo orden."),
                             (200, 420, "Piedra, hierro, diamante, obsidiana."),
                             (420, 700, "Nether, fortaleza, End. Este es el listón.")])

    # 5. generation zero
    population_scene(w, fmt, snaps[0][:77], 4200, 420,
                     "GENERACIÓN 0", "el mismo mundo para todas", "GEN 0", stride=2,
                     captions=[(0, 200, "192 moscas, el mismo mundo, el mismo cableado. "
                                        "Solo cambia cómo enchufan sus sentidos a sus patas."),
                               (200, 420, "Al principio casi ninguna sabe ni golpear un árbol.")])

    # 6. what actually learns
    card(w, fmt, [("QUÉ APRENDE Y QUÉ NO", 52, ui.DIM),
                  ("El cableado no se toca: es un dato medido.", 42, TITLE_COL),
                  ("", 16, TITLE_COL),
                  ("Evoluciona la interfaz: 1.009 parámetros.", 42, ACC),
                  ("", 16, TITLE_COL),
                  ("Y dentro de cada vida, el cuerpo pedunculado", 38, TITLE_COL),
                  ("reajusta sus propias sinapsis bajo dopamina —", 38, TITLE_COL),
                  ("el sitio exacto donde aprende una mosca real.", 38, TITLE_COL)], 9.0)

    # 7. montage
    for i, g in enumerate(snap_gens):
        if i == 0:
            continue
        population_scene(w, fmt, snaps[i][:77], 4200 + int(g), 300,
                         f"GENERACIÓN {int(g)}", "selección por truncamiento + mutación",
                         f"GEN {int(g)}", stride=3,
                         captions=[(0, 300, "Sobreviven las mejores. El resto son copias "
                                            "mutadas de ellas.")] if i == 1 else ())

    # 8. the curve
    chart_scene(w, fmt, s["history"], 10.0, hold=2.5)

    # 9. many attempts at once, then the best of them
    card(w, fmt, [("Y AHORA, A INTENTARLO", 60, ACC),
                  (f"{s['attempts']} copias de la misma mosca, en 8 mundos.", 40, TITLE_COL),
                  ("Mismo genoma, mismas sinapsis.", 42, TITLE_COL),
                  ("Lo único que las separa es el ruido neuronal.", 38, ui.DIM)], 6.0)
    population_scene(w, fmt, np.repeat(best_genome[None, :], 77, axis=0),
                     s["world_seed"], 560,
                     "LA MISMA MOSCA, 128 VECES", "el ruido las separa",
                     "INTENTOS", stride=2,
                     captions=[(0, 260, "Un cerebro real es ruidoso, y ese ruido basta "
                                        "para que cada copia juegue distinto."),
                               (260, 560, "Nos quedamos con la que llegue más lejos, "
                                          "más rápido.")])
    card(w, fmt, [("LA MEJOR CARRERA", 64, ACC),
                  (f"mundo {s['world_seed']}, mosca #{s['agent']}", 40, TITLE_COL)], 3.5)
    hero_scene(w, fmt, best_genome, s["world_seed"], s["agent"], 1400, s["attempts"],
               stride=1, hold_end=2.5,
               title="RUN FINAL", sub=f"mundo {s['world_seed']} · mosca #{s['agent']}",
               captions=[(0, 120, "Madera."),
                         (200, 320, "Pico de piedra."),
                         (500, 620, "Hierro, y luego diamante."),
                         (800, 920, "Obsidiana: el portal."),
                         (1050, 1180, "El Nether.")])

    # 9b. the part that does work: the End fight, started at the End
    eg = s.get("endgame")
    if eg:
        card(w, fmt, [("Y AHORA LA TRAMPA HONESTA", 52, ui.WARN),
                      ("Esa carrera no llega al final.", 40, TITLE_COL),
                      ("", 14, TITLE_COL),
                      ("Pero si dejamos a la misma mosca directamente", 38, TITLE_COL),
                      ("en el End, con el equipo puesto:", 38, TITLE_COL)], 6.5)
        hero_scene(w, fmt, best_genome, int(eg["world_seed"]), int(eg["agent"]), 320,
                   s["attempts"], stride=1, repeat=2, hold_end=2.5,
                   start_stage=s["endgame_stage"],
                   title="EL END", sub="arrancando desde el portal, no desde cero",
                   hud_label="END FIGHT",
                   captions=[(0, 90, "Sin la cadena de 19 pasos por delante, el circuito "
                                     "de persecución y ataque sí funciona."),
                             (90, 320, "El cuerpo pedunculado sigue reajustando sus "
                                       "sinapsis mientras pelea.")])

    # 10. result
    card(w, fmt, _result_lines(s), 5.0,
         foot=f"población {s['pop']} · {s['generations']} generaciones · "
              f"{s['genome']} parámetros libres por mosca")

    # 11. the honest part
    card(w, fmt, [("LO QUE ESTO NO ES", 52, ui.WARN),
                  ("No es Minecraft: es un sandbox con sus mecánicas.", 34, TITLE_COL),
                  ("No es el conectoma entero: es su núcleo sensoriomotor.", 34, TITLE_COL),
                  ("No son neuronas de espigas: son tasas de disparo.", 34, TITLE_COL),
                  ("Y no se lo pasan de una sentada: la ruta escrita", 34, TITLE_COL),
                  (f"a mano tarda {ui.format_time(s['scripted_ticks'])}; ellas se quedan a medias.",
                   34, TITLE_COL),
                  ("", 14, TITLE_COL),
                  ("Lo que sí es real: el grafo, los signos,", 34, ACC),
                  ("y el sitio donde ocurre el aprendizaje.", 34, ACC)], 10.0,
         foot="código y datos: repositorio vsim")
    w.close()
    build_mod.ACTIVE_SUBS = None
    subs.write(out.with_suffix(".srt"))
    return silent


# --------------------------------------------------------------- vertical cuts
def reels(genomes_path: Path, out: Path, s: dict) -> Path:
    fmt = PORTRAIT
    z = np.load(genomes_path)
    best_genome, snaps = champion(genomes_path, z), z["snapshots"]
    silent = out.with_name(out.stem + "_silent.mp4")
    w = VideoWriter(silent, fmt.size, FPS)
    build_mod.ACTIVE_SUBS = subs = Subtitles(FPS)

    hero_scene(w, fmt, best_genome, s["world_seed"], s["agent"], 210, s["attempts"],
               stride=1, title="UNA MOSCA JUEGA A MINECRAFT",
               sub="conectoma real · MaleCNS v1.0",
               captions=[(0, 105, "Esto es un cerebro de mosca de verdad."),
                         (105, 210, "Medido sinapsis a sinapsis. Publicado hace días.")])
    card(w, fmt, [("166.000 NEURONAS", 64, ACC),
                  ("125 MILLONES", 64, TITLE_COL),
                  ("DE SINAPSIS", 64, TITLE_COL)], 2.5,
         foot="Janelia FlyEM + Google Research, Cell 2026")
    population_scene(w, fmt, snaps[0][:70], 4200, 360,
                     "192 MOSCAS A LA VEZ", "mismo mundo, mismo cableado", "GEN 0",
                     stride=3,
                     captions=[(0, 360, "Las ponemos todas a jugar en paralelo.")])
    population_scene(w, fmt, snaps[-1][:70], 4200, 360,
                     f"GENERACIÓN {int(z['snapshot_gens'][-1])}", "después de evolucionar",
                     "EVOLVED", stride=3,
                     captions=[(0, 360, "Solo evoluciona cómo conectan sus sentidos con "
                                        "sus patas. El cableado no se toca.")])
    hero_scene(w, fmt, best_genome, s["world_seed"], s["agent"], 1400, s["attempts"],
               stride=3, hold_end=1.5, title="LA MEJOR CARRERA",
               sub=f"{s['attempts']} intentos en paralelo",
               captions=[(0, 300, "La mejor de todas: madera, piedra, hierro, diamante."),
                         (900, 1400, "Se queda a medias de la ruta de 19 pasos.")])
    eg = s.get("endgame")
    if eg:
        hero_scene(w, fmt, best_genome, int(eg["world_seed"]), int(eg["agent"]), 320,
                   s["attempts"], stride=1, repeat=2, hold_end=2.0,
                   start_stage=s["endgame_stage"], title="EL END",
                   sub="arrancando desde el portal", hud_label="END FIGHT",
                   captions=[(0, 320, "Pero puesta directamente en el End, con el equipo "
                                      "puesto, sí mata al dragón.")])
    card(w, fmt, _result_lines(s, small=True), 3.5, foot="repositorio: vsim")
    w.close()
    build_mod.ACTIVE_SUBS = None
    subs.write(out.with_suffix(".srt"))
    return silent


def tiktok(genomes_path: Path, out: Path, s: dict) -> Path:
    fmt = PORTRAIT
    z = np.load(genomes_path)
    best_genome, snaps = champion(genomes_path, z), z["snapshots"]
    silent = out.with_name(out.stem + "_silent.mp4")
    w = VideoWriter(silent, fmt.size, FPS)
    build_mod.ACTIVE_SUBS = subs = Subtitles(FPS)

    hero_scene(w, fmt, best_genome, s["world_seed"], s["agent"], 120, s["attempts"],
               stride=1, title="CEREBRO DE MOSCA vs MINECRAFT",
               sub="conectoma real, no una metáfora",
               captions=[(0, 120, "Cerebro de mosca real. Jugando a Minecraft.")])
    population_scene(w, fmt, snaps[-1][:70], 4200, 300, "192 A LA VEZ",
                     "hasta que una se lo pasa", "PARALELO", stride=3,
                     captions=[(0, 300, "Cientos de copias intentándolo en paralelo.")])
    hero_scene(w, fmt, best_genome, s["world_seed"], s["agent"], 700, s["attempts"],
               stride=4, hold_end=1.0, title="LA MEJOR CARRERA",
               sub="ruta Any%",
               captions=[(0, 700, "17.966 neuronas medidas, jugando.")])
    eg = s.get("endgame")
    if eg:
        hero_scene(w, fmt, best_genome, int(eg["world_seed"]), int(eg["agent"]), 320,
                   s["attempts"], stride=1, repeat=2, hold_end=1.5,
                   start_stage=s["endgame_stage"], title="EL END",
                   sub="arrancando desde el portal", hud_label="END FIGHT",
                   captions=[(0, 320, "Puesta en el End, mata al dragón.")])
    card(w, fmt, _result_lines(s, small=True), 3.0, foot="vsim")
    w.close()
    build_mod.ACTIVE_SUBS = None
    subs.write(out.with_suffix(".srt"))
    return silent


# ------------------------------------------------------------------- soundtrack
def soundtrack(seconds: float, path: Path, style: str = "long") -> Path:
    if style == "long":
        plan = [("intro", 16), ("calm", 20), ("build", 30), ("drive", 60),
                ("triumph", 30), ("calm", 20)]
    elif style == "reels":
        plan = [("build", 12), ("drive", 40), ("triumph", 20)]
    else:
        plan = [("drive", 30), ("triumph", 16)]
    total = sum(d for _, d in plan)
    scale = max(1.0, seconds / total) * 1.05
    plan = [(m, d * scale) for m, d in plan]
    return audio.build(plan, path)
