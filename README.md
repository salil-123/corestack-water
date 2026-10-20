# Fortnightly Water Detection at 10 m

## Contents

1. Purpose and scope
2. Summary of results
3. The pipeline at a glance
4. Data
5. Method
6. Parameters
7. Evaluation
8. Findings
9. Add-on: layering with existing datasets
10. Placement in the LULC hierarchy
11. Limitations
12. Using it
13. Glossary

## 1. Purpose and scope

This document describes our water detection model: a classifier that calls every 10 m pixel water or
not water once a fortnight, over any date range, followed by a temporal correction over the whole
range. It covers the data, the model, the correction, the output, how the method was evaluated and
the results.

The model answers a question existing land-cover maps do not: not only where water is, but **when**,
fortnight by fortnight, through the monsoon. Each fortnight it uses Sentinel-1 radar and Sentinel-2
optical imagery together where the sky is clear, and falls back to Sentinel-1 radar alone where cloud
hides the ground, so the monsoon months, when water matters most and optical imagery is blind, are
still mapped. It needs nothing but Sentinel-1 and Sentinel-2, and runs entirely inside Earth Engine.

Section 9 then shows an optional add-on: layering the model with existing datasets (Dynamic World and
IndiaSAT) to raise its precision further.

## 2. Summary of results

The model with temporal correction, against Dynamic World and the IndiaSAT v3 LULC map used as they
are. June 2023 to May 2024, on ground truth independent of the model's training data (Section 7.1):
three water assets and six land classes, every one counted once. Precision here is balanced precision;
plain precision and 95% intervals are in Section 7.2. Like any evaluation it is bounded by the ground
truth available: the evaluation polygons are undated, so whole-season timing is not scored against
them (the per-date check is the classifier's held-out score in Section 5.3).

| | precision | recall | false water on land | F1 | recall, seasonal water | recall, perennial water | recall, small water (GT_BINARY) |
|---|---|---|---|---|---|---|---|
| **our model** | 0.87 | **0.79** | 11.6% | **0.83** | **0.78** | 0.83 | **0.77** |
| Dynamic World | 0.83 | 0.77 | 16.0% | 0.80 | 0.70 | **0.94** | 0.66 |
| IndiaSAT v3 map | **0.99** | 0.34 | **0.3%** | 0.51 | 0.39 | 0.48 | 0.17 |

- **More water than Dynamic World, with over a quarter less false water.** Recall 0.79 against 0.77,
  false water 11.6% against 16.0%, F1 0.83 against 0.80.
- **Best on small and seasonal water.** Recall 0.77 on GT_BINARY's small ponds against Dynamic
  World's 0.66 and IndiaSAT's 0.17, and 0.78 on seasonal water against 0.70 and 0.39.
- **Cleaner in towns than Dynamic World.** False water on built-up land 6.2%, against Dynamic World's
  10%, from Sentinel alone (Section 5.3).
- **Through the monsoon.** Where cloud hides the optical imagery the model switches to radar alone.
  Under monsoon cloud this is worth +0.18 F1 over an optical model with no fallback (Section 5.2).
- **Timing.** Every pixel gets a call every fortnight, so the output gives the weeks of water and when
  a body fills and dries, not just whether it was water in the year.

## 3. The pipeline at a glance

```
boundary + start date + end date
  -> 14-day slots from the start date, plus 2 padding slots on each side
  -> per slot: Sentinel-1 median (VV, VH) and cloud-masked Sentinel-2 median
  -> per slot and pixel: clear sky -> radar + optical model; cloud -> radar-only model;
                          no radar pass -> optical-only model
  -> per slot and pixel: water if the pixel model or the neighbourhood model says so
  -> over the whole series: gap fill, then majority of 3
  -> trim the padding
  -> fortnight raster (1 / 0 / 255) and weeks of water
```

Everything up to the export runs inside Earth Engine; the only output is the raster.

## 4. Data

### 4.1 Inputs at run time

| dataset | Earth Engine ID | role | resolution |
|---|---|---|---|
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` (IW mode, VV and VH) | classifier input; sees through cloud | 10 m |
| Sentinel-2 surface reflectance | `COPERNICUS/S2_SR_HARMONIZED` | classifier input where the sky is clear | 10 m |

The add-on in Section 9 also uses Dynamic World and the IndiaSAT v3 LULC map.

### 4.2 Training labels

The classifier was trained on the seasonal water ground truth from the flood-mapping pipelines
(`seasonal_water.geojson`): **dated** polygons of 205 water bodies across India, each outlined as water or not water on specific dates, 876 polygons on 459 dates between
2016 and 2025 (about four dates per body). Because each polygon has a date, the label is what the
body looked like that fortnight, which is what the model has to learn.

From these, 8,715 pixels from 198 bodies were sampled (3,951 water, 4,764 not water), each with the
Sentinel features of its own date:

- **Interior pixels** of each polygon take the polygon's label: water, or not water on a dry date.
- **Ring pixels**, 15 to 45 m outside each water polygon on the same date, are not water. They are
  the bank, the field and the road right next to the water, the land the model is most likely to
  confuse with it.

Bodies range from under 500 m² to several km², spread from 69.7° to 91.6° E and 9.5° to 31.3° N.

### 4.3 Evaluation ground truth

Described in Section 7.1. The water bodies of one evaluation asset (IndiaSat) largely coincide with the
training bodies, so that asset's water points are left out of the scores.

## 5. Method

### 5.1 Fortnight slots and composites

**Slots.** The range is cut into 14-day slots from the start date. A leftover under 7 days joins the
last slot; a longer one is a slot of its own. Each band is named with its dates.

**Sentinel-1.** The median VV and VH of every IW-mode scene within ±7 days of the slot's centre.

**Sentinel-2.** Every scene in the slot, with cloudy pixels masked using the scene classification band
(SCL classes 1, 3, 8, 9, 10, 11: saturated, cloud shadow, medium and high cloud, cirrus, snow), then
the median of what is left. A pixel has a **clear optical look** if B3, B8 and B11 survive the masking.

**Bands derived per slot:**

| band | formula |
|---|---|
| VV/VH | VV − VH (in dB) |
| NDWI | (B3 − B8) / (B3 + B8) |
| MNDWI | (B3 − B11) / (B3 + B11) |
| NDVI | (B8 − B4) / (B8 + B4) |

### 5.2 Sensor choice: radar under cloud

Optical imagery sees water clearly but not through cloud, and cloud is heaviest in exactly the months
when water changes most. Radar sees through cloud but is coarser about small water. So the model is
chosen per fortnight and per pixel by what is actually available:

| Sentinel-1 pass | clear Sentinel-2 look | model used |
|---|---|---|
| yes | yes | fusion: radar + optical |
| yes | no (cloud) | radar only |
| no | yes | optical only |
| no | no | no data (255); filled by the temporal correction |

**How often it falls back.** Over Jalpaiguri district in 2024, a clear optical look existed on 78% of
the district-fortnights, but only 57% through kharif, and on 0.9% in the first fortnight of July. The
model fell back to radar alone on 22% of district-fortnights over the year and 43% in kharif.

**What the fallback is worth.** On the training labels, with optical withheld as the monsoon would
withhold it (39% of observations):

| | F1 | precision | recall |
|---|---|---|---|
| optical-based model, no fallback | 0.67 | 0.85 | 0.55 |
| radar only, always | 0.80 | 0.80 | 0.80 |
| **sensor choice (ours)** | **0.84** | **0.83** | **0.86** |

Without a fallback, a clouded fortnight is simply missed (recall 0.55). Radar alone everywhere throws
away the optical detail clear fortnights offer. Choosing per fortnight beats both: +0.18 F1 over no
fallback under monsoon cloud, and +0.02 in an ordinary year's cloud.

### 5.3 The classifier

**Features.**

| model | per-pixel bands | added in the neighbourhood model (3x3 and 5x5 means) |
|---|---|---|
| fusion | VV, VH, VV/VH, B3, B4, B8, B8A, B11, NDWI, MNDWI, NDVI | VV, VH, NDWI, MNDWI |
| radar only | VV, VH, VV/VH | VV, VH |
| optical only | B3, B4, B8, B8A, B11, NDWI, MNDWI, NDVI | NDWI, MNDWI |

**Pixel and neighbourhood models.** Each case has a per-pixel model and a neighbourhood model that also
sees the 3x3 and 5x5 means around the pixel. A pixel is water if either says so. The neighbourhood
model lifts ponds under 1,000 m² from 0.38 to 0.56 F1 on regions and years the model never saw; keeping
the per-pixel model alongside it stops narrow river channels from being lost.

**Model form.** Every model is a standardised linear SVM (scikit-learn `LinearSVC`, C = 1, balanced
class weights). For storage the scaling is folded into the weights, so each model is a list of bands,
weights and an intercept, and the rule is

```
water  if  w1·band1 + w2·band2 + ... + b0 > 0
```

That is plain band arithmetic, so a whole district is classified inside Earth Engine with no imagery
downloaded.

**No land-cover input.** Until 21 October 2026 the model also set every pixel ESA WorldCover maps as
built-up to not water, because roofs and roads are where radar confuses itself. Measured on the
independent ground truth (Section 7) that rule bought little: without it recall is unchanged, false
water on land rises from 10.6% to 11.6% and F1 falls from 0.835 to 0.831; false water on built-up land
rises from 0.8% to 6.2%, still below Dynamic World's 10%. So it was dropped and the model needs only
Sentinel-1 and Sentinel-2. It can be switched back on for town-heavy areas (`mask_built=True`,
Section 6).

**Held-out scores of the classifier** (training labels of Section 4.2, whole bodies held out):

| model | F1, bodies held out | F1, unseen region and unseen year | F1, ponds under 1,000 m², unseen region and year |
|---|---|---|---|
| fusion | 0.88 | 0.88 | 0.56 |
| optical only | 0.87 | 0.86 | 0.57 |
| radar only | 0.80 | 0.79 | 0.07 |

Radar alone cannot resolve the smallest ponds at 10 m; that is why optical is used whenever the sky
allows.

### 5.4 Temporal correction

The fortnight series for the whole range is corrected in one pass:

1. **Gap fill.** A fortnight with no data takes the last state seen. Fortnights before the first
   observation take the first one.
2. **Majority of three.** Each fortnight takes the majority of itself and its two neighbours.

Worked example, one pixel, 1 = water, 0 = not, · = no data:

```
classifier   0 0 1 0 0 · · 1 1 1 0 1 1 1 0 0
gap filled   0 0 1 0 0 0 0 1 1 1 0 1 1 1 0 0
corrected    0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0
```

The lone wet fortnight (a radar pass that read sand, shadow or a rain-wet field as water) is removed,
the one-fortnight dry gap inside the wet spell is filled, and the spell itself is left as the
classifier called it.

**Any date range.** The range can start and end on any date and can cross years (for example
November 2023 to August 2024); it is corrected as one series with no seam at the year boundary.

**Edges.** The range is built with two extra slots on each side, corrected, then trimmed, so the first
and last fortnights are corrected with real neighbours. On a test box whose range started in August,
the first fortnight had no data at all, and without the padding 18% of the wet pixels came out
different at the edges.

**Why the correction is light.** A wider window decides how long water lasts: a majority of 7 deletes
every wet spell under about 8 weeks and fills dry gaps of up to 6. It does not pay either. Wider
windows trade water for precision at no gain in F1, and the water they lose is the seasonal water:

| window | precision | recall | false water | F1 | recall, seasonal water |
|---|---|---|---|---|---|
| **majority of 3** | 0.87 | **0.79** | 11.6% | **0.831** | **0.78** |
| majority of 5 | 0.93 | 0.75 | 5.9% | 0.830 | 0.74 |
| majority of 7 | 0.95 | 0.73 | 3.6% | 0.825 | 0.70 |

So the correction only removes one-fortnight flicker, and the timing of real water stays the
classifier's. Even so it takes
the model from 31.0% false water to 11.6%, for 0.08 of recall (Section 7.3).

### 5.5 Output

| item | description |
|---|---|
| fortnight raster | one band per fortnight, named with its dates, e.g. `f_20231101_20231114` |
| values | `1` water, `0` not water, `255` never observed in the whole range |
| weeks of water | `water_weeks` = 2 per wet fortnight, with `observed_weeks` alongside |
| grid | 10 m, EPSG:4326, clipped to the boundary |
| delivery | Earth Engine export to Google Drive or to an Earth Engine asset |

Water-body polygons intersected with the fortnight raster give each body's water / not-water series
for the range.

## 6. Parameters

| parameter | default | effect of changing it |
|---|---|---|
| slot length | 14 days | fixed; matches the Sentinel-1 revisit over India |
| Sentinel-1 window | ±7 days around the slot centre | |
| correction window (`window`) | 3 slots | 5 or 7 give a cleaner map but delete spells under 6 or 8 weeks; `None` = no correction |
| padding (`pad`) | 2 slots per side | must be at least half the window |
| built-up rule (`mask_built`) | off | on sets ESA WorldCover built-up to not water: cleaner towns (0.8% against 6.2% false water on built-up land), adds WorldCover as an input |
| layer (`mask`) | see Section 9 | `False` = the model alone |

## 7. Evaluation

### 7.1 Ground truth

Points were sampled inside hand-labelled polygons from five Earth Engine assets, at most eight per
polygon. The IndiaSat asset is a set of labelled polygons; it is not the IndiaSAT v3 map.

| asset | Earth Engine ID | polygons | points: water | points: land | used for |
|---|---|---|---|---|---|
| GTSeasonal | `projects/ee-mtpictd/assets/GTSeasonal` | 16 | 122 | 0 | recall |
| GTPerennial | `projects/ee-mtpictd/assets/GTPerennial` | 13 | 90 | 0 | recall |
| GT_BINARY | `projects/ee-vatsal/assets/GT_BINARY_LATEST` | 288 | 581 | 584 | recall, false water |
| FarmForest | `projects/ee-indiasat/assets/Polygon_Groundtruth/FarmForest_Groundtruth` | 255 | 0 | 2,040 (farm 880, forest 1,160) | false water |
| IndiaSat | `projects/ee-indiasat/assets/IndiaSat` | 1,873 | 896 (not scored) | 13,822 (built-up 10,862, barren 2,208, greenery 752) | false water |

**Kept apart from training.** The water bodies of the IndiaSat asset largely coincide with the training
bodies of Section 4.2: 99 of its 112 water polygons overlap a training body, and 606 of its 896 water
points fall inside one. Its water points are therefore left out of every score; its land points, and
every point of the other four assets, share nothing with the training data. That leaves 793 water
points and 16,446 land points.

Scored period: June 2023 to May 2024. A point counts as water if it is wet in at least one fortnight
after correction. Dynamic World counts as water if any scene in the year labels it water; the
IndiaSAT v3 map if its 2023-24 class is 2, 3 or 4 (water in kharif, kharif and rabi, or all three
seasons).

### 7.2 Metrics

- **Recall:** the share of water points called water, averaged over the three water assets so each
  asset counts once.
- **False water:** the share of land points called water, averaged over the six land classes
  (built-up, barren, greenery, farm, forest, GT_BINARY non-water). Without this balancing, built-up's
  10,862 points would decide every number.
- **Precision:** recall / (recall + false water), the precision if water and land were equally common
  (balanced precision). Plain precision on the points as they are, where water is 4.6% and most land
  points are built-up, is lower: 0.32 for our model, 0.23 for Dynamic World, 0.88 for the IndiaSAT v3
  map.
- **F1:** from that precision and recall.
- **Uncertainty:** 95% intervals from 2,000 bootstrap draws of whole polygons within each asset and
  class (from the project's evaluation runs): our recall 0.79 [0.71, 0.86], false water 11.6% [10.1, 13.2],
  F1 0.83 [0.78, 0.87]; Dynamic World F1 0.80 [0.74, 0.84]. Our F1 is ahead of Dynamic World's in 96%
  of draws, but the interval of the difference (−0.007 to 0.073) touches zero.
- **Stricter rule:** counting a point as water only if wet in two or more fortnights gives F1 0.833
  and false water 10.1%, so the result does not hang on single wet fortnights.
- **Setting:** urban if at least a quarter of a 500 m ring around the point is WorldCover built-up,
  forest if at least half is tree cover, otherwise rural.

### 7.3 Results

The headline table is in Section 2. The effect of the correction:

| | precision | recall | false water on land | F1 |
|---|---|---|---|---|
| classifier, no correction | 0.74 | 0.87 | 31.0% | 0.80 |
| **classifier + correction** | **0.87** | **0.79** | **11.6%** | **0.83** |

False water on each land class:

| | built-up | barren | greenery | farm | forest | GT_BINARY non-water |
|---|---|---|---|---|---|---|
| **our model** | **6.2%** | **8.6%** | **13.0%** | **18.9%** | **8.7%** | **14.0%** |
| Dynamic World | 10.0% | 8.6% | 28.7% | 24.3% | 9.7% | 14.6% |
| IndiaSAT v3 map | 0.0% | 0.7% | 0.1% | 0.7% | 0.0% | 0.0% |

By setting (recall on the 695 rural and 86 forest water points; the independent set has only 12
urban water points, too few to score):

| | false water, urban | false water, rural | false water, forest | recall, rural | recall, forest |
|---|---|---|---|---|---|
| **our model** | **6.0%** | **10.4%** | 13.3% | **0.78** | **0.80** |
| Dynamic World | 9.4% | 17.6% | 9.7% | 0.73 | 0.56 |
| IndiaSAT v3 map | 0.0% | 0.3% | 0.6% | 0.25 | 0.17 |

## 8. Findings

1. **The model beats both existing maps on F1** (0.83 against 0.80 and 0.51), finding more water than
   Dynamic World with over a quarter less false water, from Sentinel-1 and Sentinel-2 alone.
2. **It finds the water the others miss.** Small and seasonal water is where existing yearly maps are
   weakest, and where a fortnightly model with radar is strongest (GT_BINARY recall 0.77 against
   0.66 and 0.17; seasonal water 0.78 against 0.70 and 0.39).
3. **It keeps mapping through the monsoon.** Falling back to radar where cloud hides the optical
   imagery is worth +0.18 F1 under monsoon cloud; without it, recall under that cloud falls to 0.55.
4. **It gives timing no yearly map gives:** water per fortnight and weeks of water per pixel, over any
   date range.
5. **The temporal correction is what makes it usable:** false water from 31.0% to 11.6%.
6. **Its weak spot is false water on vegetated land and fields** (greenery 13%, farm 19%, forest
   areas 13%), from wet fields in the monsoon and from slopes and shadow that radar reads as water;
   in towns it is 6% (roofs and roads), against Dynamic World's 9%.
   That is what the add-on in Section 9 addresses.

## 9. Add-on: layering with existing datasets

The model can also be layered with existing land-cover datasets for even higher precision. Existing
maps are very precise about **where** water can be; the model then says **when** it is there. The
model runs as before and its water is kept only inside a where layer.

### 9.1 The where layer

A pixel is kept as water only if **Dynamic World or the IndiaSAT v3 map saw water there** in the same
year.

| dataset | Earth Engine ID | used as |
|---|---|---|
| Dynamic World V1 | `GOOGLE/DYNAMICWORLD/V1` | water in any scene of the year |
| IndiaSAT v3 LULC | `projects/corestack-datasets/assets/datasets/LULC_v3_river_basin/pan_india_lulc_v3_<y>_<y+1>` | classes 2, 3, 4 |

On our ground truth every IndiaSAT water point is also Dynamic World water, so the OR of the two equals
Dynamic World there; IndiaSAT is kept because India is larger than our ground truth.

### 9.2 Results

Same model and correction in every row; only the layer changes. Same independent ground truth as
Section 7.

| layer | precision | recall | false water | F1 | recall, small water (GT_BINARY) |
|---|---|---|---|---|---|
| none (the model alone) | 0.87 | **0.79** | 11.6% | 0.83 | **0.77** |
| IndiaSAT v3 only | **0.99** | 0.34 | **0.3%** | 0.51 | 0.17 |
| Dynamic World or IndiaSAT, same year | 0.94 | 0.70 | 4.8% | 0.80 | 0.61 |
| Dynamic World or IndiaSAT, this year or the 4 before, within 20 m of a stream (the runner's `--layer both`, Section 12.4) | 0.92 | 0.76 | 6.5% | **0.83** | 0.71 |

False water on each land class with the same-year Dynamic World or IndiaSAT layer: built-up 1.2%,
barren 3.1%, greenery 8.5%, farm 11.6%, forest 0.3%, GT_BINARY non-water 3.9%.

By setting:

| layer | false water, urban | false water, rural | false water, forest | recall, rural | recall, forest |
|---|---|---|---|---|---|
| none | 6.0% | 10.4% | 13.3% | **0.78** | **0.80** |
| IndiaSAT v3 only | 0.0% | 0.3% | 0.6% | 0.25 | 0.17 |
| Dynamic World or IndiaSAT, same year | 1.1% | 5.2% | 2.3% | 0.67 | 0.55 |

- **It buys precision.** With the same-year Dynamic World or IndiaSAT layer, false water falls from
  11.6% to 4.8%, in towns from 6.0% to 1.1% and in forest areas from 13.3% to 2.3%; precision rises
  from 0.87 to 0.94.
- **It costs recall.** Recall falls from 0.79 to 0.70, and on small ponds from 0.77 to 0.61: water the
  existing maps did not see that year is removed with the rest. Overall F1 moves from 0.83 to 0.80.
  Looking back four years and adding the stream check (what the runner's layer does) keeps more of it:
  recall 0.76, false water 6.5%, F1 0.83.
- **IndiaSAT alone is too narrow a layer:** it caps recall at its own 0.34.
- So the layer suits uses where false water costs more than missed water. Where small and seasonal
  water matters most, use the model alone.

### 9.3 On the map

Twelve boxes of at least 1.5 km around IndiaSat ground-truth water polygons (4 urban, 4 rural,
4 forest). These bodies are among the model's training bodies (Section 7.1), so the maps show how the
layers behave around a body, not an independent score. They were made with the built-up rule on
(Section 5.3); without it the urban boxes show some more wet pixels on roofs and roads. Share of the polygon called water / wet share
of the rest of the box / lone wet pixels, mean by setting:

| layer | urban | rural | forest |
|---|---|---|---|
| none | 83% / 6.7% / 31 | 89% / 25.2% / 138 | 95% / 9.4% / 46 |
| IndiaSAT v3 only | 60% / 3.5% / 1 | 69% / 10.9% / 5 | 52% / 4.8% / 1 |
| Dynamic World or IndiaSAT | 83% / 5.9% / 11 | 89% / 21.7% / 28 | 95% / 6.8% / 2 |

The model keeps 83-95% of each body, as much as Dynamic World and far more than IndiaSAT; a layer
mostly removes isolated wet pixels around it. "Rest of the box" includes real rivers and ponds the
ground truth does not outline, and fields flooded in the monsoon.

## 10. Placement in the LULC hierarchy

The base LULC water class says where water is over the year. This model sits under it as a refinement
that says when. In the CoRE stack LULC app (https://core-stack.org/) it is a model in the zoo, "Water by fortnight", with 26
classes, fortnight1 to fortnight26 of the June to May year: each water pixel gets the last fortnight it
held water (fortnight26 is water all year). Used for the Water class, the classes can be renamed, and
giving several the same name merges them, for example kharif (1 to 11), rabi (12 to 19) and zaid (20 to
26). Water pixels that never held water that year stay unlabelled.

## 11. Limitations

- The evaluation polygons are undated; a polygon labelled water may have been dry in 2023-24.
- At 10 m and two-week steps, ponds under about 500 m² and spells shorter than a fortnight are below
  what the model can resolve, and radar alone cannot resolve ponds under 1,000 m².
- The training labels have about four dates per body, so the timing of fill and dry is learnt from
  sparse snapshots.
- What false water remains is mostly fields flooded in the monsoon (farm points 19% with the model
  alone, 12% with the Dynamic World or IndiaSAT layer): standing water on land labelled farm. In towns,
  roofs and roads give 6% false water without the built-up rule (Section 5.3).
- The independent water points come from three assets (793 points); they hold only 12 urban water
  points and 86 in forest, and the urban / rural / forest split rests on WorldCover 2021 and a 500 m
  ring.
- The IndiaSat asset's water bodies overlap the training bodies and are left out of the scores
  (Section 7.1); its land points are used. The IndiaSat asset and the IndiaSAT v3 map come from the same
  group, so the v3 map's false water on that asset may be lower than elsewhere.

## 12. Using it

### 12.1 Setup

1. Python 3.10 or newer: `pip install -r requirements.txt`.
2. Earth Engine access: run `earthengine authenticate` once, and set `EE_PROJECT` to an Earth Engine
   enabled Google Cloud project (or pass `--project`). On a server, point `EE_SERVICE_ACCOUNT_KEY` at
   a service account key file instead.

Nothing else is downloaded: the imagery, the layers and the model all run inside Earth Engine, and only
the finished GeoTIFF comes back.

### 12.2 What is in this repository

| file | what it is |
|---|---|
| `water.py` | the whole method in one file, as a command and as functions |
| `model.json` | the trained model: three sensor cases (`fusion`, `s1`, `s2`), each with a `pixel` and a `neighbourhood` linear model (band list, weights, intercept, scaling folded in) and its held-out scores |
| `water_raster.ipynb` | the same for a district or a boundary file, with a quick look before exporting |

### 12.3 Features the model needs

Each fortnight, for every pixel, from Sentinel-1 and cloud-masked Sentinel-2 (Section 5.1):

| sensor case | bands read by the per-pixel model | extra bands read by the neighbourhood model |
|---|---|---|
| fusion (radar + clear optical) | VV, VH, VV_VH_ratio, B3, B4, B8, B8A, B11, NDWI, MNDWI, NDVI | mean3_VV, mean3_VH, mean3_NDWI, mean3_MNDWI, mean5_VV, mean5_VH, mean5_NDWI, mean5_MNDWI |
| radar only | VV, VH, VV_VH_ratio | mean3_VV, mean3_VH, mean5_VV, mean5_VH |
| optical only | B3, B4, B8, B8A, B11, NDWI, MNDWI, NDVI | mean3_NDWI, mean3_MNDWI, mean5_NDWI, mean5_MNDWI |

`mean3_` and `mean5_` are the 3x3 and 5x5 pixel means of that band. `water.py` builds all of them; the
user only gives an area and two dates.

### 12.4 Running it

```
python water.py --bbox 77.26 28.56 77.30 28.60 --start 2023-06-01 --end 2024-05-31
python water.py --district Jalpaiguri --start 2023-06-01 --end 2024-05-31 --layer both
python water.py --boundary my_area.geojson --start 2023-11-01 --end 2024-08-31 --weeks-only
```

| option | meaning |
|---|---|
| `--bbox W S E N`, `--district NAME [--state NAME]` or `--boundary FILE` | the area (districts from FAO GAUL; a boundary file needs `geopandas`) |
| `--start`, `--end` | any two dates, may cross years |
| `--layer` | `none` (default, the model alone: best for small and seasonal water), `dw`, `indiasat` or `both`: keep water only where that map saw water in the range or the 4 years before, within 20 m of the nearest stream (Section 9) |
| `--model` | another model json, e.g. a retrained one (default `model.json`) |
| `--weeks-only` | write only `water_weeks` and `observed_weeks` |
| `--built-up-rule` | also set ESA WorldCover built-up to not water (Section 5.3) |
| `--out`, `--drive`, `--project` | output name; always export to Drive; Earth Engine project |

In Python:

```python
import water
ee = water.ee_init()
region = ee.Geometry.Rectangle([77.26, 28.56, 77.30, 28.60])
img = water.water(ee, region, "2023-06-01", "2024-05-31", layer="none")
weeks = water.weeks_of_water(img)
water.save(ee, img, region, "yamuna_2023")
```

### 12.5 Output

A GeoTIFF on a 10 m grid in EPSG:4326, clipped to the area (Section 5.5):

- default: one band per fortnight named by its dates (`f_20230601_20230614`), `1` water, `0` not
  water, `255` never observed in the whole range;
- with `--weeks-only`: `water_weeks` (2 per wet fortnight) and `observed_weeks`.

Areas under about 25 km² are downloaded to the current folder at once, with the band names inside the
file; larger ones (a district) start an Earth Engine export into Google Drive under `corestack_water/`,
which finishes in the Earth Engine Tasks tab. Either way `<name>_bands.txt` lists the bands in order.

### 12.6 Checks

`water.py` was checked against the pipeline the scores above were computed with: on a Yamuna box, for
every layer, with and without the built-up rule, for weeks of water and for a range crossing years, not
one pixel differs.

## 13. Glossary

| term | meaning |
|---|---|
| fortnight, slot | a 14-day window counted from the start date |
| clear optical look | a Sentinel-2 value for B3, B8 and B11 survives cloud masking in that slot |
| fusion / radar only / optical only | the three classifiers, picked per slot and pixel by the data available |
| neighbourhood model | a classifier that also sees the 3x3 and 5x5 means around the pixel |
| temporal correction | gap fill, then the majority of each fortnight and its two neighbours |
| padding | extra slots built either side of the range so its edges are corrected properly |
| false water | land called water, as a share of land points |
| where layer | the optional mask of Section 9 that limits where water may be called |
| GT | ground truth |
