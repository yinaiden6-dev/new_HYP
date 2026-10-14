"""Add frozen banks, full candidate-pair similarities and reproduction code to B03/F03.

Complete matrices are for selected candidate pairs, not the Cartesian product of
all 1117 queries and 5405 references. Full banks and full-gallery score caches
are included for later full-gallery experiments. No new causal experiment here.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BANK_PATHS = dict(photos='cache/image_embeddings/colnomic7b/photos.pt',
                  references='cache/image_embeddings/colnomic7b/references.pt')
SCORE_PATHS = dict(image='cache/scores/colnomic7b/image.pt', bm25='cache/scores/bm25/bm25.pt',
                  fusion='cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt')
PREDICTION_PATHS = dict(image='results/colnomic7b/image/predictions.jsonl',
    bm25='results/bm25/bm25/predictions.jsonl',
    fusion='results/bm25_image_fusion/colnomic7b/alpha_0.5/predictions.jsonl')
CODE_FILES = ['scripts/analyze_colnomic_tokens.py', 'scripts/report_colnomic_tokens.py',
    'scripts/report_colnomic_b03_tokens.py', 'scripts/encode_colnomic_images.py',
    'scripts/supplement_colnomic_archives.py', 'scripts/replay_colnomic_bundle.py',
    'evaluation/__init__.py', 'drugpack/__init__.py', 'drugpack/method/__init__.py',
    'evaluation/encoders.py', 'evaluation/scoring.py', 'evaluation/data.py',
    'evaluation/metrics.py', 'drugpack/method/token_layout.py',
    'tests/test_colnomic_token_analysis.py', 'tests/test_colnomic_archive_supplement.py']


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() +
        json.dumps(list(value.shape), separators=(',', ':')).encode() +
        value.view(torch.uint8).numpy().tobytes()).hexdigest()


def rank_candidates(scores, ids, hashes, positive_ids):
    """Match formal ranking: stable sort first, then deduplicate by image SHA."""
    positives = {hashes[ids.index(identifier)] for identifier in positive_ids}
    seen, ranked = set(), []
    for index in np.argsort(-np.asarray(scores), kind='stable'):
        if hashes[index] in seen:
            continue
        seen.add(hashes[index])
        ranked.append(int(index))
    wrong = next(index for index in ranked if hashes[index] not in positives)
    top = ranked[0]
    return dict(top1=ids[top], top1_correct=hashes[top] in positives,
        strongest_wrong=ids[wrong], strongest_wrong_rank=ranked.index(wrong)+1,
        positive_rank=min(k+1 for k, index in enumerate(ranked) if hashes[index] in positives))


def dense_similarity(query, reference):
    """All stored rows, including zero/template rows; masks are separate."""
    matrix = (query.float() @ reference.float().T).numpy()
    qi = np.flatnonzero(query.float().norm(dim=1).numpy() > 1e-8)
    ri = np.flatnonzero(reference.float().norm(dim=1).numpy() > 1e-8)
    if not len(qi) or not len(ri):
        raise ValueError('No effective tokens')
    valid = matrix[np.ix_(qi, ri)]
    winner = valid.argmax(axis=1)
    maximum = valid[np.arange(len(qi)), winner]
    return matrix, qi, ri[winner], maximum


class Assets:
    def __init__(self, root):
        self.root = root
        load = lambda path: torch.load(root/path, map_location='cpu', weights_only=True, mmap=True)
        self.banks = {population:load(path) for population, path in BANK_PATHS.items()}
        self.scores = {name:load(path) for name, path in SCORE_PATHS.items()}
        self.predictions = {name:{record['photo_id']:record for record in
            map(json.loads, (root/path).read_text().splitlines())} for name, path in PREDICTION_PATHS.items()}
        self.indices = {pop:{identifier:i for i, identifier in enumerate(bank['ids'])}
                        for pop, bank in self.banks.items()}
        self.ocr = {pop:read(root/f'cache/ocr/{pop}.json') for pop in self.banks}
        for score in self.scores.values():
            assert score['query_ids'] == self.banks['photos']['ids']
            assert score['reference_ids'] == self.banks['references']['ids']
        active = read(root/'cache/content_box_ce_polish18/deployment/active.json')
        self.deployment_path = Path(active['path'])/'manifest.json'
        self.deployment = read(self.deployment_path)
        self.deployment_queries = {row['id']:row for row in self.deployment['queries']}
        self.boxes = read(root/'dataset/annotations/subject_boxes.json')['records']
        self.ambiguities = list(csv.DictReader((root/'wrong/身份歧义/cases.csv').open()))
        self.splits = {row['photo_id']:split for split in ['train','val','test']
            for row in csv.DictReader((root/f'dataset/lists/{split}.csv').open())}
        self.hash_cache = {}
        self.metadata = self.token_metadata()
        self.ledger = self.candidate_ledger()

    def hash(self, path):
        path = Path(path)
        if path not in self.hash_cache:
            self.hash_cache[path] = file_hash(path)
        return self.hash_cache[path]

    def token_metadata(self):
        from colpali_engine.models import ColQwen2_5_Processor
        from drugpack.method.token_layout import token_layout
        processor = ColQwen2_5_Processor.from_pretrained(self.root/'weights/colnomic7b/adapter', local_files_only=True)
        result = {}
        for pop, bank in self.banks.items():
            rows = []
            for i, identifier in enumerate(bank['ids']):
                ocr = self.ocr[pop][identifier]
                assert ocr['image_sha256'] == bank['sha256'][i]
                width, height = ocr['stored_size']
                layout = token_layout(processor, 'colnomic7b', width, height)
                value = bank['embeddings'][i]
                assert value.dtype == torch.float16 and tuple(value.shape) == (layout['total'], 128), identifier
                assert torch.isfinite(value).all(), identifier
                valid = torch.where(value.float().norm(dim=1) > 1e-8)[0].tolist()
                vector_hash = token_hash(value)
                if pop == 'photos':
                    spec = self.deployment_queries[identifier]
                    assert spec['layout'] == layout and spec['all_tokens_sha256'] == vector_hash
                rows.append(dict(id=identifier, bank_index=i, original_path=bank['paths'][i],
                    image_sha256=bank['sha256'][i], raw_width=width, raw_height=height,
                    display_width=ocr['width'], display_height=ocr['height'],
                    exif_orientation=ocr['exif_orientation'], layout=layout,
                    shape=list(value.shape), dtype=str(value.dtype), all_tokens_sha256=vector_hash,
                    valid_token_indices=valid, zero_token_indices=sorted(set(range(len(value)))-set(valid)),
                    valid_nonvisual_tokens=sum(not layout['start'] <= j < layout['start']+layout['count'] for j in valid)))
            result[pop] = rows
            print(f'Token metadata verified: {pop} {len(rows)} records', flush=True)
        return result

    def candidate_ledger(self):
        qb, rb = self.banks['photos'], self.banks['references']
        rindex = self.indices['references']
        ledger = []
        ambiguous = {(row['photo_id'], row['correct_reference_sha256'], row['selected_reference_sha256'])
                     for row in self.ambiguities}
        for qi, identifier in enumerate(qb['ids']):
            ip, fp = [self.predictions[name][identifier] for name in ['image','fusion']]
            assert ip['positive_reference_ids'] == fp['positive_reference_ids']
            positives = ip['positive_reference_ids']
            row = dict(photo_id=identifier, package_id=ip['package_id'], group_id=ip['group_id'],
                split=self.splits[identifier], positive_reference_ids=positives,
                positive_reference_sha256=[rb['sha256'][rindex[p]] for p in positives])
            for name, pred in [('image', ip), ('fusion', fp)]:
                ranking = rank_candidates(self.scores[name]['scores'][qi].numpy(), rb['ids'], rb['sha256'], positives)
                assert ranking['top1'] == pred['selected_reference_id'], identifier
                assert ranking['top1_correct'] == pred['correct'] and ranking['positive_rank'] == pred['rank']
                row.update({name+'_'+key:value for key,value in ranking.items()})
                wrong = ranking['strongest_wrong']
                row[name+'_strongest_wrong_identity_ambiguity'] = any(
                    (identifier, positive_sha, rb['sha256'][rindex[wrong]]) in ambiguous
                    for positive_sha in row['positive_reference_sha256'])
            candidates = list(dict.fromkeys(positives + [row['image_top1'], row['fusion_top1'],
                row['image_strongest_wrong'], row['fusion_strongest_wrong']]))
            row['candidates'] = [dict(reference_id=rid, reference_sha256=rb['sha256'][rindex[rid]],
                is_positive=rb['sha256'][rindex[rid]] in row['positive_reference_sha256'],
                **{name+'_score':float(score['scores'][qi, rindex[rid]]) for name,score in self.scores.items()})
                for rid in candidates]
            ledger.append(row)
        print(f'Candidate ledger verified against image and fusion predictions: {len(ledger)} queries', flush=True)
        return ledger


def csv_ledger(rows):
    return [{key:json.dumps(value, ensure_ascii=False) if isinstance(value, (list,dict)) else value
             for key,value in row.items()} for row in rows]


def copy_asset(assets, source, destination, relative, files, expected_hash=None):
    source, destination = Path(source), Path(destination)
    sha = assets.hash(source)
    if expected_hash is not None and sha != expected_hash:
        raise ValueError('Source hash differs: '+str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != source.stat().st_size or file_hash(destination) != sha:
        temporary = destination.with_name(destination.name+'.copy.tmp')
        shutil.copyfile(source, temporary)
        assert file_hash(temporary) == sha
        temporary.replace(destination)
    files.append(dict(path=relative, bytes=destination.stat().st_size, sha256=sha))


def supplement(root, configuration, assets):
    folder = root/'results/token_mechanism_analysis'/configuration
    summary = read(folder/'summary.json')
    reproduction = folder/'reproduction'
    reproduction.mkdir(parents=True, exist_ok=True)
    source_manifest = []
    def copy(relative, source=None, expected=None):
        copy_asset(assets, source or root/relative, reproduction/relative,
                   'reproduction/'+relative, source_manifest, expected)
    for relative in list(BANK_PATHS.values())+list(SCORE_PATHS.values())+list(PREDICTION_PATHS.values()):
        copy(relative)
        print(configuration, 'copied', relative, flush=True)
    for path in sorted((root/'dataset/lists').glob('*.csv')):
        copy(str(path.relative_to(root)))
    with (root/'dataset/lists/packages.csv').open() as stream:
        for package in csv.DictReader(stream):
            copy('dataset/'+package['metadata_path'])
    for relative in ['dataset/lists/protocol.json','dataset/annotations/subject_boxes.json',
                     'dataset/annotations/confusion_metadata.json', 'dataset/review/error_analysis/annotations.json',
                     'cache/ocr/photos.json', 'cache/ocr/references.json', 'configs/baselines.json',
                     'wrong/身份歧义/cases.csv']:
        if (root/relative).is_file():
            copy(relative)
    for configuration_name, mode in [('B03','image'), ('F03','bm25_image_fusion__alpha_0.5')]:
        copy(f'wrong/{configuration_name}__colnomic7b__{mode}/errors.csv')
    for relative in ['cache/image_embeddings/manifests/photos.csv','cache/image_embeddings/manifests/references.csv',
                     'cache/image_embeddings/metadata.json']:
        if (root/relative).is_file():
            copy(relative)
    for path in sorted((root/'weights/colnomic7b/adapter').iterdir()):
        if path.is_file() and path.suffix in ['.json','.txt','.yml','.md']:
            copy(str(path.relative_to(root)))
    for name in ['config.json','generation_config.json','model.safetensors.index.json','README.md']:
        copy('weights/colnomic7b/base/'+name)
    copy('cache/content_box_ce_polish18/deployment/frozen/manifest.json', assets.deployment_path,
         summary['sources']['manifest_sha256'])
    # The analysis script reads this relative path from its process cwd; document cd reproduction.
    write_json(reproduction/'cache/content_box_ce_polish18/deployment/active.json',
        dict(path='cache/content_box_ce_polish18/deployment/frozen',
             original_manifest_sha256=assets.hash(assets.deployment_path), relocated_for_bundle=True))
    write_json(reproduction/'token_metadata.json', assets.metadata)
    write_json(reproduction/'candidate_ledger_all1117.json', assets.ledger)
    write_csv(reproduction/'candidate_ledger_all1117.csv', csv_ledger(assets.ledger))
    queries = sorted({case['photo_id'] for case in summary['cases']})
    target_ids = {target['photo_id'] for target in summary['targets']}
    selected_ledger = [dict(row, analysis_role='target' if row['photo_id'] in target_ids else 'success_control')
                       for row in assets.ledger if row['photo_id'] in queries]
    write_json(reproduction/'candidate_ledger_selected.json', selected_ledger)
    write_csv(reproduction/'candidate_ledger_selected.csv', csv_ledger(selected_ledger))
    pair_relations = defaultdict(list)
    for case in summary['cases']:
        for role in ['positive','negative']:
            pair_relations[case['photo_id'], case[role]].append(dict(source='original_contrast',
                target_id=case['target'], query_role=case['role'], candidate_role=role,
                token_csv=f'{case["target"]}/{case["photo_id"]}/tokens.csv'))
    for row in selected_ledger:
        roles = [(rid, 'positive') for rid in row['positive_reference_ids']]
        roles.extend((row[name+'_'+kind], name+'_'+kind) for name in ['image','fusion']
                     for kind in ['top1','strongest_wrong'])
        for rid, role in roles:
            pair_relations[row['photo_id'], rid].append(dict(source='candidate_ledger',
                query_role=row['analysis_role'], candidate_role=role))
    reference_ids = sorted({rid for _,rid in pair_relations})
    for pop, identifiers in [('photos',queries), ('references',reference_ids)]:
        bank = assets.banks[pop]
        for count, identifier in enumerate(identifiers, 1):
            index = assets.indices[pop][identifier]
            copy(bank['paths'][index], expected=bank['sha256'][index])
            if count % 100 == 0 or count == len(identifiers):
                print(configuration, f'original {pop} copied {count}/{len(identifiers)}', flush=True)
    # Formal labels are included without promoting any existing control to same-face verified.
    face_rows = []
    for target in summary['targets']:
        for query in [target['photo_id']] + ([target['control']] if target.get('control') else []):
            face_rows.append(dict(target_id=target['photo_id'], photo_id=query,
                role='target' if query==target['photo_id'] else 'same_package_success_control',
                package_id=target['package_id'], visible_face='', visible_face_status='needs_manual_review',
                identity_status='formal_dataset_label; ambiguity flags in candidate ledger',
                same_face_as_target='', same_face_status='unverified',
                note='Existing success control is not asserted to show the same packaging face'))
    write_csv(reproduction/'visible_face_review.csv', face_rows)
    pairs, maximum_error, maximum_csv_error = [], 0., 0.
    qbank, rbank = assets.banks['photos'], assets.banks['references']
    for count, ((qid,rid), relations) in enumerate(sorted(pair_relations.items()), 1):
        qi, ri = assets.indices['photos'][qid], assets.indices['references'][rid]
        matrix, valid_q, matches, maxima = dense_similarity(qbank['embeddings'][qi], rbank['embeddings'][ri])
        path = reproduction/'similarities'/qid/(rid+'.npy')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix, allow_pickle=False)
        restored = np.load(path, mmap_mode='r', allow_pickle=False)
        if restored.dtype != np.float32 or not np.array_equal(restored, matrix):
            raise ValueError('Matrix round-trip differs')
        expected = float(assets.scores['image']['scores'][qi,ri])
        score = float(maxima.mean())
        error = abs(expected-score)
        assert error < 2e-6, (qid,rid,error)
        maximum_error = max(maximum_error, error)
        checked_csvs = set()
        for relation in relations:
            if relation['source'] != 'original_contrast':
                continue
            csvpath = relation['token_csv']
            role = relation['candidate_role']
            if (csvpath,role) in checked_csvs:
                continue
            checked_csvs.add((csvpath,role))
            with (folder/csvpath).open() as stream:
                tokens = list(csv.DictReader(stream))
            assert [int(row['q_token_index']) for row in tokens] == valid_q.tolist()
            prefix = 'pos' if role=='positive' else 'neg'
            assert [int(row[prefix+'_token_index']) for row in tokens] == matches.tolist(), (csvpath,role)
            value_field = 'c_'+role
            differences = np.abs(np.array([float(row[value_field]) for row in tokens])-maxima)
            maximum_csv_error = max(maximum_csv_error, float(differences.max()))
            assert differences.max() < 2e-6, (csvpath,role)
        pairs.append(dict(photo_id=qid, reference_id=rid, query_bank_index=qi, reference_bank_index=ri,
            matrix_file=str(path.relative_to(folder)), shape=list(matrix.shape), dtype=str(matrix.dtype),
            sha256=file_hash(path), bytes=path.stat().st_size, maxsim_mean=score,
            cached_image_score=expected, replay_absolute_error=error,
            token_index_frame='original attended sequence index (both axes)', relations=relations))
        if count % 100 == 0 or count == len(pair_relations):
            print(configuration, f'complete FP32 matrices verified {count}/{len(pair_relations)}', flush=True)
    write_json(reproduction/'pair_index.json', pairs)
    write_csv(reproduction/'pair_index.csv', csv_ledger(pairs))
    code_manifest = []
    for relative in CODE_FILES:
        copy_asset(assets, root/relative, folder/'source_code'/relative, relative, code_manifest)
    versions = {name:importlib.metadata.version(name) for name in ['torch','numpy','Pillow','transformers',
        'colpali-engine','peft','accelerate','opencv-python-headless','matplotlib','pytest',
        'qwen-vl-utils','safetensors','huggingface-hub','tokenizers']}
    (folder/'source_code/requirements-pinned.txt').write_text('\n'.join(name+'=='+version for name,version in versions.items())+'\n')
    write_json(folder/'source_code/manifest.json', dict(kind='analysis and frozen encoding code snapshot',
        dependency_versions=versions, files=code_manifest,
        validation='tests recorded separately in reproduction/validation.json; no model inference claimed'))
    model = read(root/'configs/baselines.json')['models']['colnomic7b']
    write_json(reproduction/'model_recipe.json', dict(model=model,
        image_decode='PIL Image.open(...).convert(RGB), no EXIF transpose before encoding',
        bank_storage_dtype='torch.float16', dot_product_dtype='torch.float32', renormalize_cache_vectors=False,
        score='mean_i max_j dot(q_i,g_j); valid vectors have L2 norm >1e-8',
        nonvisual_attended_tokens='included when nonzero; not stripped',
        model_weights_included=False, processor_and_tokenizer_included=True,
        encoder_gpu_recipe='original sharded recipe requires two visible CUDA GPUs and device cuda:0',
        historical_bank_provenance='preserved verbatim in original .pt banks; mixed generated and reused records',
        encoding_validation='code and processor checked; no new weight-loading or model inference run in this packaging task'))
    validation = dict(status='passed', configuration=configuration, targets=len(target_ids),
        original_contrasts=len(summary['cases']), selected_queries=len(queries), selected_references=len(reference_ids),
        full_candidate_pair_matrices=len(pairs), matrix_bytes=sum(pair['bytes'] for pair in pairs),
        all_original_token_csv_argmax_and_similarities_verified=True,
        max_matrix_to_cached_score_error=maximum_error, max_matrix_to_token_csv_error=maximum_csv_error,
        original_embedding_banks=dict(photos=len(qbank['ids']),references=len(rbank['ids'])),
        all_token_layouts_shapes_and_finiteness_verified=True,
        all_query_vectors_match_frozen_deployment_hashes=True,
        full_gallery_candidate_ledger_verified_queries=len(assets.ledger),
        copied_original_files_sha256_verified=True,
        new_model_inference_performed=False, controlled_experiments_performed=False,
        verified_utc=datetime.now(timezone.utc).isoformat())
    write_json(reproduction/'validation.json', validation)
    write_json(reproduction/'source_files.json', source_manifest)
    documentation(folder, configuration, validation)
    # Hash every supplemented file, including copied code, docs and generated indexes.
    items = []
    for base in [reproduction, folder/'source_code']:
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.name != 'FILES.sha256':
                items.append(f'{file_hash(path)}  {path.relative_to(folder).as_posix()}')
    items.extend(f'{file_hash(folder/name)}  {name}' for name in ['REPRODUCIBILITY.md','NEXT_EXPERIMENTS.md'])
    (reproduction/'FILES.sha256').write_text('\n'.join(items)+'\n')
    marker = '\n\n### 原始特征与复现补充\n\n已补充原始特征、完整候选对相似度矩阵和冻结编码代码。入口：[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。\n'
    report = folder/'REPORT.md'
    if '### 原始特征与复现补充' not in report.read_text():
        with report.open('a') as stream:
            stream.write(marker)
    print(configuration, 'SUPPLEMENT COMPLETE', json.dumps(validation,ensure_ascii=False), flush=True)
    return validation


def documentation(folder, cfg, validation):
    text = f'''# {cfg} 原始特征与复现包

本轮仅补齐归档。已有对应/贡献诊断仍然有效；没有执行几何约束重新排序、Probe 或布局干预。

- 原分析目标 {validation['targets']} 例、{validation['original_contrasts']} 组对照。
- 附带 **全部 1117 张照片、5405 行图库的原始 FP16 token embedding**，保留数值、顺序、零向量、非图像模板 token 和历史 provenance。
- 本包 {validation['selected_queries']} 张目标/成功对照、{validation['selected_references']} 个参考候选的原图；字节和 EXIF 均保留。
- **{validation['full_candidate_pair_matrices']} 个完整 FP32 token×token 相似度矩阵**，两端全部缓存行都在矩阵中，不是最大值、Top-k、热力图或量化近似。覆盖原始所有对照，以及每张选定照片的正确候选、图像/融合实际 Top-1 和图像/融合最高分错误候选。重复使用的候选对只存一次，由索引链接。
- image/BM25/fusion 的完整 [1117,5405] 分数缓存、全部候选清单、身份表、划分表、OCR 多边形、确认主体框、小型处理器/分词器配置和编码源代码。

## 文件入口

| 内容 | 路径 |
| --- | --- |
| 原始 embedding 银行 | `reproduction/cache/image_embeddings/colnomic7b/photos.pt`、`references.pt` |
| 每个 ID 的银行行、布局、有效 token 索引、EXIF、向量 SHA | `reproduction/token_metadata.json` |
| 完整候选对矩阵及原始轴索引 | `reproduction/similarities/<photo_id>/<reference_id>.npy` |
| 候选对、原始对照目录、分数、矩阵 SHA | `reproduction/pair_index.json` / `.csv` |
| 本包比较对象 / 全部 1117 张比较对象 | `reproduction/candidate_ledger_selected.json` / `candidate_ledger_all1117.json`，各附 CSV |
| 同面核实表 | `reproduction/visible_face_review.csv`（未核实，不作为已完成同面对照） |
| 矩阵数值及 CSV 复现验证 | `reproduction/validation.json` |
| 每个补充文件 SHA-256 | `reproduction/FILES.sha256` |
| 编码入口 / 原始编码实现 | `source_code/scripts/encode_colnomic_images.py` / `source_code/evaluation/encoders.py` |
| 模型版本、数据类型和预处理配方 | `reproduction/model_recipe.json` |
| 后续实验协议 | `NEXT_EXPERIMENTS.md` |

## 比较对象与评分

`image_top1` / `fusion_top1` 是正式排序实际第一名，可能是正确候选。`image_strongest_wrong` / `fusion_strongest_wrong` 是排除所有正样本图像 SHA 后的最高分候选；即使方法检索正确也保存此候选。先稳定排序，再按图像 SHA 去重；同分遵循正式参考行顺序。身份歧义仅按具体正负图像 SHA 配对转移，不能只按照片转移。

矩阵 `[i,j] = photos[qi][i].float() @ references[ri][j].float()`；没有二次归一化。用于 MaxSim 的轴通过 `valid_token_indices` 选择（范数 >1e-8）。每个有效 query token 取有效 reference token 的最大值，再取 query 均值。非零模板 token 参与原始分数。原始网格 `row_raw/col_raw` 位于未应用 EXIF 的编码帧；OCR、主体框和显示坐标属于 EXIF 转正帧，可用快照中的 `display_xy` 转换。

矩阵覆盖上述选定候选对，**没有物化全部 1117×5405 个 token 矩阵**。全图库 embedding 与分数已齐全，可按需计算任意候选对。原图只覆盖本包候选与照片；以后执行全图库几何估计还需要其余原图。所有身份标签保持数据集定义，包装面和同文字不同布局类别还需人工核实；原始 train/val/test 表不能自动代替按包装身份划分的 Probe 协议。

## 无需 GPU 的复现

解压后进入 `{cfg}` 目录。使用 `source_code/requirements-pinned.txt` 中的冻结依赖；CUDA Torch 构建需与机器适配。矩阵和特征读取、核查不需要模型权重。

```bash
python source_code/scripts/replay_colnomic_bundle.py --bundle . --verify-files --verify-matrices
PYTHONPATH=source_code python -m pytest -q source_code/tests
```

测试导入需要 `PYTHONPATH=source_code`，或先进入 `source_code` 执行 `python -m pytest -q tests`。

单个矩阵可以直接 `np.load(path, mmap_mode='r', allow_pickle=False)`。银行可用 `torch.load(path, map_location='cpu', weights_only=True, mmap=True)`；ID 的正式索引已保存在 `pair_index` 中。

重新运行原有分析（会估计 SIFT 几何并绘图，需要已附原图，不加载编码器）：

```bash
cd reproduction
python ../source_code/scripts/analyze_colnomic_tokens.py --root . --configuration {cfg} --all-errors --output ../rerun_{cfg}
```

## 对新图/编辑图重新编码

7B base 和 adapter **模型权重不在压缩包内**；完整冻结特征已在包内。权重的仓库与 revision 位于 `model_recipe.json`，需使用匹配的完整本地 checkpoint。原有分片编码配方要求两张可见 CUDA GPU，入口 `--device cuda:0`。处理器配置已经附带，`--prepare-only` 不需要模型权重或 GPU。

```bash
python source_code/scripts/encode_colnomic_images.py --root reproduction --image dataset/path/to/image.png --prepare-only --output prepared.json
python source_code/scripts/encode_colnomic_images.py --root reproduction --image /absolute/path/edited.png --adapter /absolute/path/complete_adapter --base /absolute/path/complete_base --device cuda:0 --output edited_embeddings.pt
```

也可用 `--manifest images.csv` 批量编码，CSV 包含 `id,path`，相对路径以 `--root` 为基准。脚本写入新输出；不会更新附带的冻结银行。预处理已验证，未在本轮重新加载 7B 权重进行模型推理；不同硬件重新编码不承诺逐位一致，应另报告和原始特征的数值偏差。
'''
    (folder/'REPRODUCIBILITY.md').write_text(text)
    (folder/'source_code/README.md').write_text('完整数据与运行说明见 ../REPRODUCIBILITY.md。\n\n编码入口：scripts/encode_colnomic_images.py；原始编码实现：evaluation/encoders.py。\n\n在本目录执行 python -m pytest -q tests。requirements-pinned.txt 为当前环境版本快照。\n')
    protocol = '''# 区分竞争假设：后续实验协议（尚未执行）

本轮只补齐原始数据和可运行代码。下列四阶段属于下一轮实验，不能从本包的 MaxSim 对应/贡献诊断直接宣称编码器丢失信息、空间匹配具有因果作用或通用检索已经改善。

1. 比较对象：使用 candidate_ledger 与 pair_index 固定 GT、image strongest wrong、fusion strongest wrong，同时保留实际 Top-1。人工检查可见包装面、文字内容和身份歧义；填写 visible_face_review。同身份成功照片只有面一致且可见区域相当时才作为排版机制对照。保留缺失/拒绝原因，不能把未核实的对照计入同面成功样本。
2. 冻结匹配对比：预先固定几何估计（不得用测试真值选择对齐）、失败回退规则与候选集。在训练/验证身份上选择 lambda 等参数，再锁定测试。比较原始 MaxSim、特征+几何约束、几何信息单独评分，并记录拒绝拟合比例。先同候选集 pairwise margin，再完整图库检验 Recall/MRR 和原本成功照片的损害。当前 SIFT homography 是诊断记录，不是已经验证的几何评分算法。加入坐标后改善不能证明 embedding 自身编码了布局。
3. 仅冻结特征 Probe：严格禁止输入原始坐标、token 顺序索引或网格元数据给特征-only head（这些可作独立坐标对照）。位置/行号 Probe 与同文字不同实例布局 Probe 分开；后者需要经核实的身份相关布局类别和同面样本。按 package_id/近重复设计族分组，训练/验证/测试身份不交叉，并报告文字泄漏与面混杂。加入多随机种子标签打乱、坐标-only、文本-only、多数类等对照。预测绝对位置不等于区分实例布局；失败只说明所选 Probe 在该数据规模下没有检测到信息，不能证明信息完全不存在。
4. 布局干预：使用一张可审计包装面作为底图，保持所有文字、颜色、剂量/容量与图案一致，只移动末尾词或改变换行。若存在拍摄透视，先记录统一的编辑坐标系与回投规则。使用相同编辑过程但不改变布局的 sham 对照，校验字符/OCR内容、颜色、编辑区域与分辨率；编码前后记录像素 SHA、编辑参数、模型 revision、原始 token SHA 和分数。比较 base、sham 与布局编辑的差值，报告编辑痕迹和重新编码偏差。不能仅用遮挡贡献宣称因果。

所有阈值/探针/干预决定在测试前固定；完整测试和难度子集必须同时计数改善、损害、无变化及歧义剔除口径。两包使用相同原始银行与全图库分数，避免 B03/F03 分析对象不一致。
'''
    (folder/'NEXT_EXPERIMENTS.md').write_text(protocol)


def archive(root, configuration):
    directory = root/'results/token_mechanism_analysis'
    folder = directory/configuration
    stem = ('B03_' if configuration=='B03' else '')+'token_mechanism_analysis_20261013'
    destination = directory/(stem+'.zip')
    temporary = directory/(stem+'.updated.tmp.zip')
    files = sorted(path for path in folder.rglob('*') if path.is_file())
    prefix = 'token_mechanism_analysis/'+configuration+'/'
    expected = {prefix+path.relative_to(folder).as_posix():path.stat().st_size for path in files}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipout:
        for i, path in enumerate(files, 1):
            method = zipfile.ZIP_STORED if path.suffix.lower() in ['.jpg','.jpeg','.png'] else zipfile.ZIP_DEFLATED
            zipout.write(path, prefix+path.relative_to(folder).as_posix(), compress_type=method)
            if i % 200 == 0 or i == len(files):
                print(configuration, f'archive written {i}/{len(files)} files, {temporary.stat().st_size/2**30:.2f} GiB', flush=True)
    with zipfile.ZipFile(temporary) as zipped:
        assert len(zipped.infolist()) == len(expected)
        assert {info.filename:info.file_size for info in zipped.infolist()} == expected
        for i, info in enumerate(zipped.infolist(), 1):
            # A full read verifies each ZIP CRC. Also compare supplemented files to their source bytes.
            hashed = hashlib.sha256()
            with zipped.open(info) as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024), b''):
                    hashed.update(chunk)
            relative = info.filename[len(prefix):]
            if relative.startswith(('reproduction/','source_code/')):
                assert hashed.hexdigest() == file_hash(folder/relative), relative
            if i % 200 == 0 or i == len(files):
                print(configuration, f'archive CRC verified {i}/{len(files)}', flush=True)
    sha = file_hash(temporary)
    validation = dict(status='passed', configuration=configuration, archive=str(destination),
        files=len(files), source_bytes=sum(expected.values()), archive_bytes=temporary.stat().st_size,
        sha256=sha, all_paths_and_sizes_verified=True, all_entries_crc_verified=True,
        supplemented_file_sha256_verified_inside_archive=True, archive_excluded_from_itself=True,
        reproduction_validation=read(folder/'reproduction/validation.json'),
        tests=read(folder/'source_code/test_results.json') if (folder/'source_code/test_results.json').exists() else None,
        verified_utc=datetime.now(timezone.utc).isoformat())
    os.replace(temporary, destination)
    checksum = destination.with_name(destination.name+'.sha256')
    checksum.write_text(sha+'  '+destination.name+'\n')
    write_json(directory/(stem+'.validation.json'), validation)
    print(configuration, 'ARCHIVE COMPLETE', json.dumps(validation, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--configuration', choices=['F03','B03'], action='append')
    parser.add_argument('--archive-only', action='store_true')
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configurations = args.configuration or ['F03','B03']
    if not args.archive_only:
        assets = Assets(root)
        for cfg in configurations:
            supplement(root, cfg, assets)
    if not args.no_archive:
        for cfg in configurations:
            archive(root, cfg)


if __name__ == '__main__':
    main()
