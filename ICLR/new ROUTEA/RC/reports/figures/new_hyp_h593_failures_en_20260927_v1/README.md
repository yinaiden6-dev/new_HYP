# H593: all COST1 retrieval failures — English figures

All **112 incorrect predictions** from the original H593 grouped held-out COST1 model, exactly the medicine-model source used in the 90 successful-correction figures. RAW is 426/593 and COST1 is 481/593. This is the original external seven-parameter COST1, not the post-LLM model.

- 92 wrong HOLD, 17 wrong-to-wrong SWITCH, and 3 RAW-correct breaks.
- 89 have the correct reference in natural C128; 23 are candidate-recall misses. These 23 remain included. Their correct gallery photograph is shown, but no target-pair map or score was computed in the original C128 experiment. The empty panel is marked unavailable, never filled with invented weights.
- 9 DIFFICULT, 1 NDV2 and 102 OUTCOME cases; difficult/new difficult appear first. Every sealed failure is included, without selection by appearance or heatmap brightness.
- English 3200×1800 PNGs, SVGs, single-page PDFs, [112-page PDF](failures112.pdf), contact sheets, [offline HTML gallery](index.html) and [CSV index](data/cases.csv). References retain the enlarged full-column layout. Fold numbers appear only in provenance data.

The figure compares the **final incorrect reference** against the correct reference. The RAW winner and final action are separately labeled; wrong-to-wrong switches do not substitute the old RAW reference for the actual final error. Correct identity is determined by the original repaired gallery labels. Heatmaps are literal cached FP64 query-token weights, nearest-pixel display and fixed [0,1]. They are not segmentation masks.

Per-case JSON keeps the complete natural candidate axis, all 127 sealed challenger logits plus HOLD=0, original COST1 parameters, source fold and training exclusion, action replay and source hashes. NPZ files keep native query/reference maps for available pairs and the 127×6 decision feature matrix, not model embeddings. Rendering runs no encoder, matcher or training. Original experiments and the successful rescue figures are unchanged.

Source: `results/rc_six_cause_isolation_v1/loss_binding/result.json` and its five original fold payloads; H593 feature and RoMa caches provide the actual intermediate evidence. [Source bindings](source_manifest.json) and [validation](validation.json) document verification. Source photos are embedded in panels; original photos and large feature caches are separately stored and not copied into the Git figure directory.

## All cases

| No. | Query / PNG | Final error | Target in C128 | Evidence |
|---:|---|---|---|---|
| 1 | [DIFFICULT-0011](medicine/001_COST1_DIFFICULT-0011.png) | WRONG_HOLD | inside | [data](data/001_COST1_DIFFICULT-0011.json) |
| 2 | [DIFFICULT-0018](medicine/002_COST1_DIFFICULT-0018.png) | WRONG_HOLD | outside | [data](data/002_COST1_DIFFICULT-0018.json) |
| 3 | [DIFFICULT-0028](medicine/003_COST1_DIFFICULT-0028.png) | WRONG_HOLD | inside | [data](data/003_COST1_DIFFICULT-0028.json) |
| 4 | [DIFFICULT-0050](medicine/004_COST1_DIFFICULT-0050.png) | WRONG_HOLD | inside | [data](data/004_COST1_DIFFICULT-0050.json) |
| 5 | [DIFFICULT-0057](medicine/005_COST1_DIFFICULT-0057.png) | WRONG_HOLD | inside | [data](data/005_COST1_DIFFICULT-0057.json) |
| 6 | [DIFFICULT-0058](medicine/006_COST1_DIFFICULT-0058.png) | WRONG_HOLD | inside | [data](data/006_COST1_DIFFICULT-0058.json) |
| 7 | [DIFFICULT-0098](medicine/007_COST1_DIFFICULT-0098.png) | WRONG_HOLD | inside | [data](data/007_COST1_DIFFICULT-0098.json) |
| 8 | [DIFFICULT-0099](medicine/008_COST1_DIFFICULT-0099.png) | WRONG_HOLD | inside | [data](data/008_COST1_DIFFICULT-0099.json) |
| 9 | [DIFFICULT-0100](medicine/009_COST1_DIFFICULT-0100.png) | WRONG_HOLD | inside | [data](data/009_COST1_DIFFICULT-0100.json) |
| 10 | [NDV2-012-P03](medicine/010_COST1_NDV2-012-P03.png) | BREAK | inside | [data](data/010_COST1_NDV2-012-P03.json) |
| 11 | [OUTCOME-0001](medicine/011_COST1_OUTCOME-0001.png) | WRONG_HOLD | inside | [data](data/011_COST1_OUTCOME-0001.json) |
| 12 | [OUTCOME-0090](medicine/012_COST1_OUTCOME-0090.png) | WRONG_HOLD | inside | [data](data/012_COST1_OUTCOME-0090.json) |
| 13 | [OUTCOME-0093](medicine/013_COST1_OUTCOME-0093.png) | WRONG_SWITCH | inside | [data](data/013_COST1_OUTCOME-0093.json) |
| 14 | [OUTCOME-0096](medicine/014_COST1_OUTCOME-0096.png) | WRONG_SWITCH | inside | [data](data/014_COST1_OUTCOME-0096.json) |
| 15 | [OUTCOME-0097](medicine/015_COST1_OUTCOME-0097.png) | WRONG_HOLD | inside | [data](data/015_COST1_OUTCOME-0097.json) |
| 16 | [OUTCOME-0098](medicine/016_COST1_OUTCOME-0098.png) | WRONG_HOLD | inside | [data](data/016_COST1_OUTCOME-0098.json) |
| 17 | [OUTCOME-0100](medicine/017_COST1_OUTCOME-0100.png) | WRONG_HOLD | outside | [data](data/017_COST1_OUTCOME-0100.json) |
| 18 | [OUTCOME-0104](medicine/018_COST1_OUTCOME-0104.png) | WRONG_HOLD | inside | [data](data/018_COST1_OUTCOME-0104.json) |
| 19 | [OUTCOME-0105](medicine/019_COST1_OUTCOME-0105.png) | WRONG_HOLD | inside | [data](data/019_COST1_OUTCOME-0105.json) |
| 20 | [OUTCOME-0106](medicine/020_COST1_OUTCOME-0106.png) | WRONG_HOLD | inside | [data](data/020_COST1_OUTCOME-0106.json) |
| 21 | [OUTCOME-0107](medicine/021_COST1_OUTCOME-0107.png) | WRONG_HOLD | inside | [data](data/021_COST1_OUTCOME-0107.json) |
| 22 | [OUTCOME-0110](medicine/022_COST1_OUTCOME-0110.png) | WRONG_HOLD | inside | [data](data/022_COST1_OUTCOME-0110.json) |
| 23 | [OUTCOME-0132](medicine/023_COST1_OUTCOME-0132.png) | WRONG_HOLD | inside | [data](data/023_COST1_OUTCOME-0132.json) |
| 24 | [OUTCOME-0133](medicine/024_COST1_OUTCOME-0133.png) | WRONG_HOLD | inside | [data](data/024_COST1_OUTCOME-0133.json) |
| 25 | [OUTCOME-0134](medicine/025_COST1_OUTCOME-0134.png) | WRONG_SWITCH | inside | [data](data/025_COST1_OUTCOME-0134.json) |
| 26 | [OUTCOME-0138](medicine/026_COST1_OUTCOME-0138.png) | WRONG_SWITCH | inside | [data](data/026_COST1_OUTCOME-0138.json) |
| 27 | [OUTCOME-0140](medicine/027_COST1_OUTCOME-0140.png) | WRONG_HOLD | inside | [data](data/027_COST1_OUTCOME-0140.json) |
| 28 | [OUTCOME-0141](medicine/028_COST1_OUTCOME-0141.png) | WRONG_HOLD | inside | [data](data/028_COST1_OUTCOME-0141.json) |
| 29 | [OUTCOME-0150](medicine/029_COST1_OUTCOME-0150.png) | WRONG_HOLD | inside | [data](data/029_COST1_OUTCOME-0150.json) |
| 30 | [OUTCOME-0172](medicine/030_COST1_OUTCOME-0172.png) | WRONG_HOLD | outside | [data](data/030_COST1_OUTCOME-0172.json) |
| 31 | [OUTCOME-0198](medicine/031_COST1_OUTCOME-0198.png) | WRONG_HOLD | outside | [data](data/031_COST1_OUTCOME-0198.json) |
| 32 | [OUTCOME-0199](medicine/032_COST1_OUTCOME-0199.png) | WRONG_HOLD | outside | [data](data/032_COST1_OUTCOME-0199.json) |
| 33 | [OUTCOME-0203](medicine/033_COST1_OUTCOME-0203.png) | WRONG_HOLD | outside | [data](data/033_COST1_OUTCOME-0203.json) |
| 34 | [OUTCOME-0205](medicine/034_COST1_OUTCOME-0205.png) | WRONG_HOLD | outside | [data](data/034_COST1_OUTCOME-0205.json) |
| 35 | [OUTCOME-0207](medicine/035_COST1_OUTCOME-0207.png) | WRONG_HOLD | outside | [data](data/035_COST1_OUTCOME-0207.json) |
| 36 | [OUTCOME-0208](medicine/036_COST1_OUTCOME-0208.png) | WRONG_SWITCH | outside | [data](data/036_COST1_OUTCOME-0208.json) |
| 37 | [OUTCOME-0209](medicine/037_COST1_OUTCOME-0209.png) | WRONG_HOLD | outside | [data](data/037_COST1_OUTCOME-0209.json) |
| 38 | [OUTCOME-0213](medicine/038_COST1_OUTCOME-0213.png) | WRONG_HOLD | inside | [data](data/038_COST1_OUTCOME-0213.json) |
| 39 | [OUTCOME-0270](medicine/039_COST1_OUTCOME-0270.png) | WRONG_HOLD | inside | [data](data/039_COST1_OUTCOME-0270.json) |
| 40 | [OUTCOME-0271](medicine/040_COST1_OUTCOME-0271.png) | WRONG_HOLD | inside | [data](data/040_COST1_OUTCOME-0271.json) |
| 41 | [OUTCOME-0273](medicine/041_COST1_OUTCOME-0273.png) | WRONG_SWITCH | outside | [data](data/041_COST1_OUTCOME-0273.json) |
| 42 | [OUTCOME-0274](medicine/042_COST1_OUTCOME-0274.png) | WRONG_HOLD | outside | [data](data/042_COST1_OUTCOME-0274.json) |
| 43 | [OUTCOME-0278](medicine/043_COST1_OUTCOME-0278.png) | WRONG_HOLD | inside | [data](data/043_COST1_OUTCOME-0278.json) |
| 44 | [OUTCOME-0281](medicine/044_COST1_OUTCOME-0281.png) | WRONG_HOLD | outside | [data](data/044_COST1_OUTCOME-0281.json) |
| 45 | [OUTCOME-0282](medicine/045_COST1_OUTCOME-0282.png) | WRONG_HOLD | outside | [data](data/045_COST1_OUTCOME-0282.json) |
| 46 | [OUTCOME-0331](medicine/046_COST1_OUTCOME-0331.png) | WRONG_HOLD | inside | [data](data/046_COST1_OUTCOME-0331.json) |
| 47 | [OUTCOME-0334](medicine/047_COST1_OUTCOME-0334.png) | BREAK | inside | [data](data/047_COST1_OUTCOME-0334.json) |
| 48 | [OUTCOME-0337](medicine/048_COST1_OUTCOME-0337.png) | BREAK | inside | [data](data/048_COST1_OUTCOME-0337.json) |
| 49 | [OUTCOME-0338](medicine/049_COST1_OUTCOME-0338.png) | WRONG_HOLD | outside | [data](data/049_COST1_OUTCOME-0338.json) |
| 50 | [OUTCOME-0341](medicine/050_COST1_OUTCOME-0341.png) | WRONG_HOLD | inside | [data](data/050_COST1_OUTCOME-0341.json) |
| 51 | [OUTCOME-0347](medicine/051_COST1_OUTCOME-0347.png) | WRONG_HOLD | inside | [data](data/051_COST1_OUTCOME-0347.json) |
| 52 | [OUTCOME-0348](medicine/052_COST1_OUTCOME-0348.png) | WRONG_HOLD | inside | [data](data/052_COST1_OUTCOME-0348.json) |
| 53 | [OUTCOME-0373](medicine/053_COST1_OUTCOME-0373.png) | WRONG_SWITCH | inside | [data](data/053_COST1_OUTCOME-0373.json) |
| 54 | [OUTCOME-0379](medicine/054_COST1_OUTCOME-0379.png) | WRONG_HOLD | inside | [data](data/054_COST1_OUTCOME-0379.json) |
| 55 | [OUTCOME-0382](medicine/055_COST1_OUTCOME-0382.png) | WRONG_SWITCH | inside | [data](data/055_COST1_OUTCOME-0382.json) |
| 56 | [OUTCOME-0389](medicine/056_COST1_OUTCOME-0389.png) | WRONG_HOLD | inside | [data](data/056_COST1_OUTCOME-0389.json) |
| 57 | [OUTCOME-0405](medicine/057_COST1_OUTCOME-0405.png) | WRONG_SWITCH | outside | [data](data/057_COST1_OUTCOME-0405.json) |
| 58 | [OUTCOME-0406](medicine/058_COST1_OUTCOME-0406.png) | WRONG_SWITCH | outside | [data](data/058_COST1_OUTCOME-0406.json) |
| 59 | [OUTCOME-0407](medicine/059_COST1_OUTCOME-0407.png) | WRONG_SWITCH | inside | [data](data/059_COST1_OUTCOME-0407.json) |
| 60 | [OUTCOME-0411](medicine/060_COST1_OUTCOME-0411.png) | WRONG_SWITCH | outside | [data](data/060_COST1_OUTCOME-0411.json) |
| 61 | [OUTCOME-0412](medicine/061_COST1_OUTCOME-0412.png) | WRONG_HOLD | outside | [data](data/061_COST1_OUTCOME-0412.json) |
| 62 | [OUTCOME-0413](medicine/062_COST1_OUTCOME-0413.png) | WRONG_HOLD | outside | [data](data/062_COST1_OUTCOME-0413.json) |
| 63 | [OUTCOME-0416](medicine/063_COST1_OUTCOME-0416.png) | WRONG_SWITCH | inside | [data](data/063_COST1_OUTCOME-0416.json) |
| 64 | [OUTCOME-0434](medicine/064_COST1_OUTCOME-0434.png) | WRONG_HOLD | inside | [data](data/064_COST1_OUTCOME-0434.json) |
| 65 | [OUTCOME-0435](medicine/065_COST1_OUTCOME-0435.png) | WRONG_HOLD | inside | [data](data/065_COST1_OUTCOME-0435.json) |
| 66 | [OUTCOME-0439](medicine/066_COST1_OUTCOME-0439.png) | WRONG_HOLD | inside | [data](data/066_COST1_OUTCOME-0439.json) |
| 67 | [OUTCOME-0441](medicine/067_COST1_OUTCOME-0441.png) | WRONG_SWITCH | inside | [data](data/067_COST1_OUTCOME-0441.json) |
| 68 | [OUTCOME-0445](medicine/068_COST1_OUTCOME-0445.png) | WRONG_HOLD | inside | [data](data/068_COST1_OUTCOME-0445.json) |
| 69 | [OUTCOME-0446](medicine/069_COST1_OUTCOME-0446.png) | WRONG_HOLD | inside | [data](data/069_COST1_OUTCOME-0446.json) |
| 70 | [OUTCOME-0465](medicine/070_COST1_OUTCOME-0465.png) | WRONG_HOLD | outside | [data](data/070_COST1_OUTCOME-0465.json) |
| 71 | [OUTCOME-0468](medicine/071_COST1_OUTCOME-0468.png) | WRONG_SWITCH | inside | [data](data/071_COST1_OUTCOME-0468.json) |
| 72 | [OUTCOME-0473](medicine/072_COST1_OUTCOME-0473.png) | WRONG_HOLD | outside | [data](data/072_COST1_OUTCOME-0473.json) |
| 73 | [OUTCOME-0474](medicine/073_COST1_OUTCOME-0474.png) | WRONG_HOLD | inside | [data](data/073_COST1_OUTCOME-0474.json) |
| 74 | [OUTCOME-0475](medicine/074_COST1_OUTCOME-0475.png) | WRONG_HOLD | inside | [data](data/074_COST1_OUTCOME-0475.json) |
| 75 | [OUTCOME-0477](medicine/075_COST1_OUTCOME-0477.png) | WRONG_SWITCH | inside | [data](data/075_COST1_OUTCOME-0477.json) |
| 76 | [OUTCOME-0529](medicine/076_COST1_OUTCOME-0529.png) | WRONG_HOLD | inside | [data](data/076_COST1_OUTCOME-0529.json) |
| 77 | [OUTCOME-0530](medicine/077_COST1_OUTCOME-0530.png) | WRONG_HOLD | inside | [data](data/077_COST1_OUTCOME-0530.json) |
| 78 | [OUTCOME-0531](medicine/078_COST1_OUTCOME-0531.png) | WRONG_HOLD | inside | [data](data/078_COST1_OUTCOME-0531.json) |
| 79 | [OUTCOME-0532](medicine/079_COST1_OUTCOME-0532.png) | WRONG_HOLD | inside | [data](data/079_COST1_OUTCOME-0532.json) |
| 80 | [OUTCOME-0533](medicine/080_COST1_OUTCOME-0533.png) | WRONG_HOLD | inside | [data](data/080_COST1_OUTCOME-0533.json) |
| 81 | [OUTCOME-0534](medicine/081_COST1_OUTCOME-0534.png) | WRONG_HOLD | inside | [data](data/081_COST1_OUTCOME-0534.json) |
| 82 | [OUTCOME-0536](medicine/082_COST1_OUTCOME-0536.png) | WRONG_HOLD | inside | [data](data/082_COST1_OUTCOME-0536.json) |
| 83 | [OUTCOME-0538](medicine/083_COST1_OUTCOME-0538.png) | WRONG_HOLD | inside | [data](data/083_COST1_OUTCOME-0538.json) |
| 84 | [OUTCOME-0555](medicine/084_COST1_OUTCOME-0555.png) | WRONG_HOLD | inside | [data](data/084_COST1_OUTCOME-0555.json) |
| 85 | [OUTCOME-0557](medicine/085_COST1_OUTCOME-0557.png) | WRONG_HOLD | inside | [data](data/085_COST1_OUTCOME-0557.json) |
| 86 | [OUTCOME-0576](medicine/086_COST1_OUTCOME-0576.png) | WRONG_HOLD | inside | [data](data/086_COST1_OUTCOME-0576.json) |
| 87 | [OUTCOME-0588](medicine/087_COST1_OUTCOME-0588.png) | WRONG_HOLD | inside | [data](data/087_COST1_OUTCOME-0588.json) |
| 88 | [OUTCOME-0598](medicine/088_COST1_OUTCOME-0598.png) | WRONG_HOLD | inside | [data](data/088_COST1_OUTCOME-0598.json) |
| 89 | [OUTCOME-0635](medicine/089_COST1_OUTCOME-0635.png) | WRONG_HOLD | inside | [data](data/089_COST1_OUTCOME-0635.json) |
| 90 | [OUTCOME-0638](medicine/090_COST1_OUTCOME-0638.png) | WRONG_HOLD | inside | [data](data/090_COST1_OUTCOME-0638.json) |
| 91 | [OUTCOME-0651](medicine/091_COST1_OUTCOME-0651.png) | WRONG_HOLD | inside | [data](data/091_COST1_OUTCOME-0651.json) |
| 92 | [OUTCOME-0661](medicine/092_COST1_OUTCOME-0661.png) | WRONG_HOLD | inside | [data](data/092_COST1_OUTCOME-0661.json) |
| 93 | [OUTCOME-0676](medicine/093_COST1_OUTCOME-0676.png) | WRONG_HOLD | inside | [data](data/093_COST1_OUTCOME-0676.json) |
| 94 | [OUTCOME-0695](medicine/094_COST1_OUTCOME-0695.png) | WRONG_HOLD | inside | [data](data/094_COST1_OUTCOME-0695.json) |
| 95 | [OUTCOME-0733](medicine/095_COST1_OUTCOME-0733.png) | WRONG_HOLD | inside | [data](data/095_COST1_OUTCOME-0733.json) |
| 96 | [OUTCOME-0742](medicine/096_COST1_OUTCOME-0742.png) | WRONG_HOLD | inside | [data](data/096_COST1_OUTCOME-0742.json) |
| 97 | [OUTCOME-0747](medicine/097_COST1_OUTCOME-0747.png) | WRONG_HOLD | inside | [data](data/097_COST1_OUTCOME-0747.json) |
| 98 | [OUTCOME-0748](medicine/098_COST1_OUTCOME-0748.png) | WRONG_HOLD | inside | [data](data/098_COST1_OUTCOME-0748.json) |
| 99 | [OUTCOME-0755](medicine/099_COST1_OUTCOME-0755.png) | WRONG_HOLD | inside | [data](data/099_COST1_OUTCOME-0755.json) |
| 100 | [OUTCOME-0769](medicine/100_COST1_OUTCOME-0769.png) | WRONG_HOLD | inside | [data](data/100_COST1_OUTCOME-0769.json) |
| 101 | [OUTCOME-0797](medicine/101_COST1_OUTCOME-0797.png) | WRONG_HOLD | inside | [data](data/101_COST1_OUTCOME-0797.json) |
| 102 | [OUTCOME-0807](medicine/102_COST1_OUTCOME-0807.png) | WRONG_HOLD | outside | [data](data/102_COST1_OUTCOME-0807.json) |
| 103 | [OUTCOME-0809](medicine/103_COST1_OUTCOME-0809.png) | WRONG_HOLD | inside | [data](data/103_COST1_OUTCOME-0809.json) |
| 104 | [OUTCOME-0810](medicine/104_COST1_OUTCOME-0810.png) | WRONG_HOLD | inside | [data](data/104_COST1_OUTCOME-0810.json) |
| 105 | [OUTCOME-0811](medicine/105_COST1_OUTCOME-0811.png) | WRONG_HOLD | inside | [data](data/105_COST1_OUTCOME-0811.json) |
| 106 | [OUTCOME-0812](medicine/106_COST1_OUTCOME-0812.png) | WRONG_HOLD | inside | [data](data/106_COST1_OUTCOME-0812.json) |
| 107 | [OUTCOME-0814](medicine/107_COST1_OUTCOME-0814.png) | WRONG_HOLD | inside | [data](data/107_COST1_OUTCOME-0814.json) |
| 108 | [OUTCOME-0815](medicine/108_COST1_OUTCOME-0815.png) | WRONG_SWITCH | inside | [data](data/108_COST1_OUTCOME-0815.json) |
| 109 | [OUTCOME-0818](medicine/109_COST1_OUTCOME-0818.png) | WRONG_HOLD | inside | [data](data/109_COST1_OUTCOME-0818.json) |
| 110 | [OUTCOME-0819](medicine/110_COST1_OUTCOME-0819.png) | WRONG_HOLD | inside | [data](data/110_COST1_OUTCOME-0819.json) |
| 111 | [OUTCOME-0820](medicine/111_COST1_OUTCOME-0820.png) | WRONG_HOLD | inside | [data](data/111_COST1_OUTCOME-0820.json) |
| 112 | [OUTCOME-0821](medicine/112_COST1_OUTCOME-0821.png) | WRONG_HOLD | inside | [data](data/112_COST1_OUTCOME-0821.json) |
