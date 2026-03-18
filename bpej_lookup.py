"""
bpej_lookup.py  –  Bonita půdy z BPEJ / bpej.vumop.cz
=======================================================
Dotazuje API VÚMOPu přes POST request s WGS-84 souřadnicemi.

Endpoint: https://bpej.vumop.cz/existBpej.php
POST parametry: coorx (longitude), coory (latitude)
Odpověď: BPEJ kód jako prostý text, např. "30700"

Funguje pouze pro ČR – mimo ČR nebo na nezem. půdě vrací fertility=None.
"""

import requests

BPEJ_URL = "https://bpej.vumop.cz/existBpej.php"
TIMEOUT  = 8

# Hranice ČR ve WGS-84
CZ_LAT_MIN, CZ_LAT_MAX = 48.55, 51.06
CZ_LON_MIN, CZ_LON_MAX = 12.09, 18.87


# ---------------------------------------------------------------------------
# BPEJ kód → složky
# ---------------------------------------------------------------------------
def _parse_bpej_code(code: str):
    """Vrátí (klimatický_region, HPJ, sklonitost, hloubka_skelet) nebo None."""
    code = str(code).strip()
    # Může přijít jako "30700" nebo "3.07.00" – normalizujeme
    code = code.replace(".", "").zfill(5)
    if len(code) != 5 or not code.isdigit():
        return None
    return int(code[0]), int(code[1:3]), int(code[3]), int(code[4])


# ---------------------------------------------------------------------------
# HPJ → třída úrodnosti  (dle vyhlášky MZe 327/1998 Sb.)
# ---------------------------------------------------------------------------
def _hpj_to_fertility(hpj: int) -> str | None:
    if hpj < 1 or hpj > 89:
        return None
    if hpj <= 13:  return "Velmi úrodná"   # černozemě, nejlepší hnědozemě
    if hpj <= 35:  return "Úrodná"          # střední hnědozemě, luvizemě, kambizemě
    return "Neúrodná"                        # oglejené, mělké, skeletovité, rašeliny


def _climate_correction(region: int, base: str) -> str:
    order = ["Neúrodná", "Úrodná", "Velmi úrodná"]
    idx = order.index(base)
    if region <= 1:   idx = min(idx + 1, 2)   # nejteplejší jižní Morava
    elif region >= 7: idx = max(idx - 1, 0)   # hory
    return order[idx]


def _slope_correction(slope: int, base: str) -> str:
    if slope < 5: return base
    order = ["Neúrodná", "Úrodná", "Velmi úrodná"]
    return order[max(order.index(base) - 1, 0)]


# ---------------------------------------------------------------------------
# HLAVNÍ FUNKCE
# ---------------------------------------------------------------------------
def get_fertility_from_coords(lat: float, lon: float) -> dict:
    """
    POST na bpej.vumop.cz/existBpej.php s coorx (lon) a coory (lat).
    Vrátí slovník:
      fertility:     'Velmi úrodná' | 'Úrodná' | 'Neúrodná' | None
      bpej_code:     '30700' | None
      bpej_decoded:  dict | None
      source:        'BPEJ/vumop' | 'mimo_CR' | 'chyba'
      error:         None | str
    """
    r = {"fertility": None, "bpej_code": None,
         "bpej_decoded": None, "source": None, "error": None}

    # Mimo ČR?
    if not (CZ_LAT_MIN <= lat <= CZ_LAT_MAX and CZ_LON_MIN <= lon <= CZ_LON_MAX):
        r["source"] = "mimo_CR"
        return r

    # POST request – stejné parametry jaké posílá bpej.vumop.cz
    try:
        resp = requests.post(
            BPEJ_URL,
            data={"coorx": str(lon), "coory": str(lat)},
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": "https://bpej.vumop.cz/",
                "Origin": "https://bpej.vumop.cz",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        r["source"] = "chyba"
        r["error"]  = "Server neodpověděl (timeout)"
        return r
    except requests.exceptions.RequestException as e:
        r["source"] = "chyba"
        r["error"]  = f"Chyba sítě: {e}"
        return r

    # Odpověď je prostý text s BPEJ kódem, např. "30700"
    raw = resp.text.strip()

    # Prázdná odpověď = bod není na zemědělské půdě
    if not raw:
        r["source"] = "chyba"
        r["error"]  = "Bod neleží na zemědělské půdě (BPEJ nenalezeno)"
        return r

    # Parsování kódu
    parsed = _parse_bpej_code(raw)
    if parsed is None:
        r["source"] = "chyba"
        r["error"]  = f"Nečekaná odpověď serveru: '{raw}'"
        return r

    climate_region, hpj, slope, depth_skel = parsed
    base = _hpj_to_fertility(hpj)
    if base is None:
        r["source"] = "chyba"
        r["error"]  = f"Neznámé HPJ číslo: {hpj} (kód: {raw})"
        return r

    fertility = _slope_correction(slope, _climate_correction(climate_region, base))

    r.update({
        "fertility":  fertility,
        "bpej_code":  raw.zfill(5),
        "bpej_decoded": {
            "klimaticky_region": climate_region,
            "hpj":               hpj,
            "sklonitost":        slope,
            "hloubka_skelet":    depth_skel,
            "base_fertility":    base,
        },
        "source": "BPEJ/vumop",
        "error":  None,
    })
    return r


# ---------------------------------------------------------------------------
# UI popis
# ---------------------------------------------------------------------------
_HPJ_DESC = {
    range(1,  4):  "Černozem modální",
    range(4,  8):  "Černozem luvická/karbonátová",
    range(8,  10): "Černozem pelická",
    range(10, 14): "Hnědozem na spraši",
    range(14, 18): "Luvizem na spraši",
    range(18, 25): "Kambizem střední",
    range(25, 36): "Kambizem mělká/skeletovitá",
    range(36, 47): "Oglejená kambizem",
    range(47, 56): "Pseudoglej",
    range(56, 67): "Glej",
    range(67, 73): "Rendzina",
    range(73, 79): "Rašelinná/organická půda",
    range(79, 90): "Ostatní/specifická půda",
}

def describe_hpj(hpj: int) -> str:
    for rng, desc in _HPJ_DESC.items():
        if hpj in rng:
            return desc
    return f"HPJ {hpj:02d}"

def format_bpej_info(result: dict, lang: str = "cs") -> str:
    if not result.get("bpej_code"):
        return ""
    d    = result["bpej_decoded"]
    soil = describe_hpj(d["hpj"])
    code = result["bpej_code"]
    # Formátuj jako X.XX.XX
    fmt  = f"{code[0]}.{code[1:3]}.{code[3:]}"
    if lang == "en":
        return (f"BPEJ: **{fmt}** | "
                f"Climate region: {d['klimaticky_region']} | "
                f"Soil: {soil} (HPJ {d['hpj']:02d}) | "
                f"Fertility: **{result['fertility']}**")
    return (f"Kód BPEJ: **{fmt}** | "
            f"Klimatický region: {d['klimaticky_region']} | "
            f"Půda: {soil} (HPJ {d['hpj']:02d}) | "
            f"Úrodnost: **{result['fertility']}**")