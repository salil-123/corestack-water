"""Fortnightly water at 10 m from Sentinel-1 and Sentinel-2, for any area and any date range.

Every 10 m pixel is called water or not water once a fortnight, then the series is cleaned over the whole
range. Everything runs inside Google Earth Engine; only the finished GeoTIFF comes back.

  slots        14-day slots from the start date (a leftover under a week joins the last one), plus two
               extra slots each side so the ends get cleaned with real neighbours
  per slot     median Sentinel-1 VV and VH within 7 days of the slot centre, and the median of the
               slot's cloud-masked Sentinel-2 scenes
  sensor       radar + a clear optical look -> fusion model, radar only (cloud) -> radar model,
               optical only (no radar pass) -> optical model, neither -> no data (255)
  models       each case has a per-pixel and a neighbourhood linear model (model.json); water if either
               says so. Linear, so it runs as band arithmetic: sum(w * band) + b > 0
  correction   unseen slots take the last seen state, then each slot takes the majority of itself and its
               two neighbours: a lone wet fortnight goes, a lone dry one inside a wet spell is filled
  layer        optional: keep water only where Dynamic World or IndiaSAT saw water in the range or the 4
               years before, on pixels within 20 m of their nearest stream (MERIT HAND)
  output       one band per fortnight (1 water, 0 not, 255 never observed), or weeks of water

    python water.py --bbox 77.26 28.56 77.30 28.60 --start 2023-06-01 --end 2024-05-31
    python water.py --district Jalpaiguri --start 2023-06-01 --end 2024-05-31 --layer both
    python water.py --boundary my_area.geojson --start 2023-11-01 --end 2024-08-31 --weeks-only

Method, evaluation and every option: README.md.
"""
import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path

MODEL = Path(__file__).resolve().parent / "model.json"
LEN = 14                         # days in a slot
PAD = 2                          # slots built either side of the range, then dropped
WINDOW = 3                       # slots in the majority vote
NODATA = 255
S1_DAYS = 7                      # Sentinel-1 scenes within this many days of the slot centre
BAD_SCL = [1, 3, 8, 9, 10, 11]   # Sentinel-2 scene classes masked out: saturated, shadow, cloud, cirrus, snow
OPTICAL = ["B3", "B4", "B8", "B8A", "B11"]

# the optional "where" layer
DW = "GOOGLE/DYNAMICWORLD/V1"
INDIASAT = "projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_{}_{}"
INDIASAT_WATER = (2, 3, 4)
INDIASAT_YEARS = range(2017, 2025)       # IndiaSAT v3 maps on Earth Engine, 2017-18 to 2024-25
LOOKBACK_YEARS = 4
HAND_MAX = 20                            # metres above the nearest stream
LAYERS = ("none", "dw", "indiasat", "both")

WORLDCOVER = "ESA/WorldCover/v200/2021"  # only for the optional built-up rule (class 50)
GAUL = "FAO/GAUL/2015/level2"            # district boundaries
SMALL_KM2 = 25                           # below this the GeoTIFF downloads straight away


# ----------------------------- setup -----------------------------
def ee_init(project=None):
    """Earth Engine on EE_PROJECT (or `project`). A service account key in EE_SERVICE_ACCOUNT_KEY is used
    if set, otherwise the credentials from `earthengine authenticate`."""
    import ee
    project = project or os.getenv("EE_PROJECT")
    key = os.getenv("EE_SERVICE_ACCOUNT_KEY")
    if key and os.path.exists(key):
        creds = ee.ServiceAccountCredentials(json.load(open(key))["client_email"], key)
        ee.Initialize(creds, project=project)
    else:
        ee.Initialize(project=project)
    return ee


def load_model(path=MODEL):
    return json.loads(Path(path).read_text())


def district(ee, name, state=None):
    """A district boundary from FAO GAUL by name. A name used in two states needs `state`."""
    fc = (ee.FeatureCollection(GAUL).filter(ee.Filter.eq("ADM0_NAME", "India"))
          .filter(ee.Filter.eq("ADM2_NAME", name)))
    if state:
        fc = fc.filter(ee.Filter.eq("ADM1_NAME", state))
    states = sorted(set(fc.aggregate_array("ADM1_NAME").getInfo()))
    if not states:
        raise ValueError(f"no district called {name!r}" + (f" in {state}" if state else ""))
    if len(states) > 1:
        raise ValueError(f"{name!r} is a district in {', '.join(states)}; pass --state")
    return fc.geometry()


def boundary(ee, path):
    """A local GeoJSON / shapefile boundary, dissolved and flattened to 2-D (needs geopandas)."""
    import geopandas as gpd
    import shapely
    g = gpd.read_file(path)
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
    shape = shapely.force_2d(g.union_all()).buffer(0)
    if shape.is_empty:
        raise ValueError(f"{path} has no usable polygon in it")
    return ee.Geometry(shape.__geo_interface__)


# ----------------------------- slots -----------------------------
def _d(x):
    return x if isinstance(x, date) else date.fromisoformat(str(x)[:10])


def slots(start, end):
    """(start, end, centre) per 14-day slot, end exclusive, covering start..end (both included)."""
    start, stop = _d(start), _d(end) + timedelta(days=1)
    out, s = [], start
    while s < stop:
        e = min(s + timedelta(days=LEN), stop)
        if (stop - e).days < 7:              # a leftover under a week joins this slot
            e = stop
        out.append((s, e, s + (e - s) / 2))
        s = e
    return out


def padded(start, end, pad=PAD):
    asked = slots(start, end)
    first, last = asked[0][0], asked[-1][1]
    before = [(first - timedelta(days=LEN * k), first - timedelta(days=LEN * (k - 1))) for k in range(pad, 0, -1)]
    after = [(last + timedelta(days=LEN * k), last + timedelta(days=LEN * (k + 1))) for k in range(pad)]
    return [(s, e, s + (e - s) / 2) for s, e in before] + asked + [(s, e, s + (e - s) / 2) for s, e in after]


def label(s, e):
    """'f_20231101_20231114': the dates are in the band name."""
    return f"f_{s:%Y%m%d}_{(e - timedelta(days=1)):%Y%m%d}"


# ----------------------------- one fortnight -----------------------------
def _empty(ee, names):
    """A fully masked image with these bands, so a slot with no scene still has them."""
    return ee.Image.constant([0] * len(names)).rename(names).updateMask(ee.Image(0))


def features(ee, region, start, end, centre):
    """The slot's bands, plus where it has a clear optical look and where it has radar."""
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
          .filterDate(ee.Date(str(centre)).advance(-S1_DAYS, "day"), ee.Date(str(centre)).advance(S1_DAYS, "day"))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]).merge(ee.ImageCollection([_empty(ee, ["VV", "VH"])])).median())

    def clear(img):
        ok = img.select("SCL").remap(BAD_SCL, [0] * len(BAD_SCL), 1)
        return img.select(OPTICAL).updateMask(ok)

    opt = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(str(start), str(end))
           .map(clear).merge(ee.ImageCollection([_empty(ee, OPTICAL)])).median())
    b3, b4, b8, b11 = (opt.select(b) for b in ("B3", "B4", "B8", "B11"))
    nd = lambda a, b, n: a.subtract(b).divide(a.add(b).add(1e-6)).rename(n)
    vv, vh = s1.select("VV"), s1.select("VH")
    img = (vv.rename("VV").addBands(vh.rename("VH")).addBands(vv.subtract(vh).rename("VV_VH_ratio"))
           .addBands(nd(b3, b8, "NDWI")).addBands(nd(b3, b11, "MNDWI")).addBands(nd(b8, b4, "NDVI"))
           .addBands(opt.select(OPTICAL)))
    return img, b3.mask().And(b8.mask()).And(b11.mask()), vv.mask().And(vh.mask())


def with_means(ee, img, bands):
    """Add the neighbourhood bands a model reads: mean3_X / mean5_X = band X averaged over the 3x3 / 5x5
    window. Taken on the masked image, so cloud and missing passes drop out of the average."""
    for k, radius in ((3, 10), (5, 20)):
        src = sorted({b.split("_", 1)[1] for b in bands if b.startswith(f"mean{k}_")})
        if src:
            m = img.select(src).reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(radius, "meters"))
            img = img.addBands(m.rename([f"mean{k}_{c}" for c in src]))
    return img


def _twins(entry):
    """A sensor case holds a pixel model and a neighbourhood model."""
    return [entry[k] for k in ("pixel", "neighbourhood") if k in entry] or [entry]


def _water(img, entry):
    """Water if either of the case's linear models says so."""
    out = None
    for m in _twins(entry):
        score = None
        for band, w in zip(m["bands"], m["weights"]):
            term = img.select(band).multiply(w)
            score = term if score is None else score.add(term)
        v = score.add(m["intercept"]).gt(0)
        out = v if out is None else out.Or(v)
    return out


def water_band(ee, region, slot, model, built_up_rule=False):
    """One fortnight: 1 water, 0 not, 255 nothing to go on."""
    img, has_opt, has_sar = features(ee, region, *slot)
    img = with_means(ee, img, {b for case in ("fusion", "s1", "s2") for m in _twins(model[case]) for b in m["bands"]})
    opt, sar, filled = has_opt.unmask(0), has_sar.unmask(0), img.unmask(0)
    verdict = (_water(filled, model["fusion"])
               .where(sar.And(opt.Not()), _water(filled, model["s1"]))
               .where(opt.And(sar.Not()), _water(filled, model["s2"])))
    if built_up_rule:            # optional: WorldCover built-up is never water (cleaner towns, one more input)
        verdict = verdict.where(ee.Image(WORLDCOVER).select("Map").eq(50).unmask(0), 0)
    return verdict.where(sar.Not().And(opt.Not()), NODATA).toUint8()


# ----------------------------- the series -----------------------------
def correct(bands, window=WINDOW):
    """Temporal correction: unseen slots take the last seen state (the first one before any sighting),
    then a `window` majority. Pixels never seen stay 255."""
    seen = [b.neq(NODATA) for b in bands]
    fwd, prev = [], None
    for b, ok in zip(bands, seen):
        prev = b if prev is None else b.where(ok.Not(), prev)
        fwd.append(prev)
    filled, nxt = [None] * len(fwd), None
    for t in range(len(fwd) - 1, -1, -1):
        nxt = fwd[t] if nxt is None else fwd[t].where(fwd[t].eq(NODATA), nxt)
        filled[t] = nxt
    never = filled[0].eq(NODATA)
    wet = [f.eq(1) for f in filled]
    h, out = window // 2, []
    for t in range(len(wet)):
        win = [wet[min(max(k, 0), len(wet) - 1)] for k in range(t - h, t + h + 1)]
        votes = win[0]
        for w in win[1:]:
            votes = votes.add(w)
        out.append(votes.gt(h).toUint8().where(never, NODATA))
    return out


def where_layer(ee, start, end, layer="both"):
    """Where water may be: Dynamic World water in any scene and/or IndiaSAT water, over the range and the
    4 years before it, and no more than 20 m above the nearest stream."""
    start, end = _d(start), _d(end)
    first = date(start.year - LOOKBACK_YEARS, start.month, min(start.day, 28 if start.month == 2 else 31))
    parts = []
    if layer in ("dw", "both"):
        parts.append(ee.ImageCollection(DW).filterDate(str(first), str(end + timedelta(days=1)))
                     .select("label").map(lambda i: i.eq(0).toByte().rename("where"))
                     .merge(ee.ImageCollection([ee.Image.constant(0).toByte().rename("where")])).max())
    if layer in ("indiasat", "both"):
        years = [y for y in INDIASAT_YEARS if date(y, 7, 1) <= end and date(y + 1, 7, 1) > first]
        parts += [ee.Image(INDIASAT.format(y, y + 1)).select("predicted_label")
                  .remap(list(INDIASAT_WATER), [1] * len(INDIASAT_WATER), 0).unmask(0).toByte().rename("where")
                  for y in years]
    mask = ee.ImageCollection(parts).max()
    return mask.And(ee.Image("MERIT/Hydro/v1_0_1").select("hnd").unmask(0).lte(HAND_MAX))


def water(ee, region, start, end, layer="none", model=None, window=WINDOW, built_up_rule=False):
    """The fortnight stack for one region and range: one band per asked slot, named by its dates.
    window=None skips the correction."""
    model = model or load_model()
    pad = PAD if window else 0
    raw = [water_band(ee, region, sl, model, built_up_rule) for sl in padded(start, end, pad)]
    bands = correct(raw, window) if window else raw
    bands = bands[pad:len(bands) - pad] if pad else bands
    if layer != "none":
        where = where_layer(ee, start, end, layer)
        bands = [b.where(where.Not().And(b.neq(NODATA)), 0) for b in bands]
    return ee.Image.cat([b.rename(label(s, e)) for b, (s, e, _) in zip(bands, slots(start, end))]).clip(region)


def weeks_of_water(stack):
    """Weeks of water (2 per wet fortnight) and weeks observed, per pixel."""
    stack = stack.select("f.*")
    return (stack.eq(1).reduce("sum").multiply(2).rename("water_weeks")
            .addBands(stack.neq(NODATA).reduce("sum").multiply(2).rename("observed_weeks")).toUint8())


def sample_weeks(stack, region, n=3000, seed=0, per_request=4):
    """A quick look without an export: weeks of water at `n` random pixels, a few fortnights per request
    (all 26 at once is too much for an interactive Earth Engine call)."""
    names = [b for b in stack.bandNames().getInfo() if b.startswith("f")]
    wet, seen = {}, {}
    for k in range(0, len(names), per_request):
        part = names[k:k + per_request]
        for f in stack.select(part).sample(region=region, scale=10, numPixels=n, seed=seed,
                                           geometries=True).getInfo()["features"]:
            key = tuple(f["geometry"]["coordinates"])
            vals = [f["properties"].get(b) for b in part]
            wet[key] = wet.get(key, 0) + 2 * sum(v == 1 for v in vals)
            seen[key] = seen.get(key, 0) + 2 * sum(v in (0, 1) for v in vals)
    return list(wet.values()), list(seen.values())


# ----------------------------- output -----------------------------
def save(ee, img, region, name, drive=False, folder="corestack_water"):
    """A GeoTIFF: downloaded here for small areas, an Earth Engine export to Google Drive for big ones.
    Either way `<name>_bands.txt` lists the bands in order (a Drive export drops their names)."""
    bands = img.bandNames().getInfo()
    Path(f"{name}_bands.txt").write_text("\n".join(bands) + "\n")
    km2 = region.area(1).getInfo() / 1e6
    if drive or km2 > SMALL_KM2:
        task = ee.batch.Export.image.toDrive(image=img, description=name, folder=folder, fileNamePrefix=name,
                                             region=region, scale=10, crs="EPSG:4326", maxPixels=1e13)
        task.start()
        print(f"{km2:.0f} km2: Drive export started ({task.id}) into {folder}/{name}.tif; see the Earth Engine Tasks tab")
        return task.id
    import requests
    url = img.getDownloadURL({"region": region, "scale": 10, "crs": "EPSG:4326", "format": "GEO_TIFF"})
    Path(f"{name}.tif").write_bytes(requests.get(url, timeout=900).content)
    try:                         # put the band names into the file too, when rasterio is around
        import rasterio
        with rasterio.open(f"{name}.tif", "r+") as dst:
            for i, b in enumerate(bands, start=1):
                dst.set_band_description(i, b)
    except ImportError:
        pass
    print(f"wrote {name}.tif ({km2:.1f} km2, {len(bands)} bands)")
    return Path(f"{name}.tif")


def main():
    ap = argparse.ArgumentParser(description="fortnightly water at 10 m from Sentinel-1 and Sentinel-2")
    area = ap.add_mutually_exclusive_group(required=True)
    area.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    area.add_argument("--district", help="district name in FAO GAUL (India)")
    area.add_argument("--boundary", help="a GeoJSON or shapefile")
    ap.add_argument("--state", help="state, when the district name exists in two states")
    ap.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last day, YYYY-MM-DD (may be in another year)")
    ap.add_argument("--layer", choices=LAYERS, default="none",
                    help="keep water only where dw, indiasat or both saw water (default none: the model alone)")
    ap.add_argument("--model", default=str(MODEL), help="model json (default model.json next to this file)")
    ap.add_argument("--weeks-only", action="store_true", help="write water_weeks and observed_weeks only")
    ap.add_argument("--built-up-rule", action="store_true", help="ESA WorldCover built-up is never water")
    ap.add_argument("--out", help="output name (default from the layer and the dates)")
    ap.add_argument("--drive", action="store_true", help="always export to Google Drive")
    ap.add_argument("--project", help="Earth Engine project (default EE_PROJECT)")
    a = ap.parse_args()

    ee = ee_init(a.project)
    region = (ee.Geometry.Rectangle(a.bbox) if a.bbox else
              district(ee, a.district, a.state) if a.district else boundary(ee, a.boundary))
    img = water(ee, region, a.start, a.end, a.layer, load_model(a.model), built_up_rule=a.built_up_rule)
    if a.weeks_only:
        img = weeks_of_water(img)
    save(ee, img, region, a.out or f"water_{a.layer}_{a.start}_{a.end}".replace("-", ""), a.drive)


if __name__ == "__main__":
    main()
