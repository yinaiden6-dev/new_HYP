# GroZi external runtime launcher repair V2

Job 5143667 RAW shard 0 completed successfully. Job 5143668 failed before RoMa model construction with ROMA_RUNTIME_PREFIX: V1 used .venv-colpali for both stages, while the unchanged RoMa source profile requires .venv-romav2. Dependent jobs 5143669, 5143670 and 5143671 were cancelled automatically. This is a launcher defect; no RoMa predictions or label join were produced.

V2 selects .venv-colpali for RAW and .venv-romav2 for RoMa, preserving the original profile's Python/module versions, TORCH_HOME and numerical loop. The worker, scoring, final heads, data manifests, candidate selection and join remain V1 and are not edited. A separate repair authority pins the original inference authority and new launcher; the launcher verifies these pins before executing the existing worker. Existing qualified RAW shard 0 is reused without modifying its payload or authority.

Resume RoMa shard 0 first, then RAW shards 1-59 with concurrency 46, then RoMa shards 1-59 with concurrency 46, then the unchanged all-480 sealed-prediction join. RAW receives 10 minutes per task; RoMa receives 15 minutes. The remaining arrays depend on pilot engineering success, not accuracy. No intermediate target label join, training, threshold change or external model selection is allowed.

Validation: import the complete worker in the required RoMa environment and call its original CPU-callable check_runtime; compare compiled numerical loop hash with V1; validate Bash syntax and execute both launcher branches against an external timeout stub to assert interpreter, stage, shard and timeout arguments. Compare actual submitted batch spools byte for byte with their launcher files.
