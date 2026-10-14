"""Read and incrementally repair the released OCR and embedding bank formats."""
import fcntl
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from .data import digest, file_hash, read_json, text_hash, write_json
from .dense import DENSE_DIMENSIONS, IMAGE_ENCODERS
from drugpack.runtime import python_executable


def save_tensor(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,temporary);temporary.replace(path)


def valid_embedding(value,model,view,has_text=True,tokens=None):
    if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
        return False
    if model in DENSE_DIMENSIONS:
        if value.shape!=(DENSE_DIMENSIONS[model],) or value.dtype!=torch.float32:
            return False
    elif value.ndim!=2 or value.shape[1]!=128 or value.dtype!=torch.float16:
        return False
    if view=='text' and not has_text:
        return not torch.count_nonzero(value) and (model in DENSE_DIMENSIONS or value.shape[0]==0)
    if view=='text' and model not in DENSE_DIMENSIONS and value.shape[0]!=tokens:
        return False
    norms=value.float().norm(dim=-1)
    norms=norms[norms>0] if norms.ndim else norms.reshape(1)
    return bool(norms.numel() and ((norms-1).abs()<0.03).all())


class Cache:
    def __init__(self,dataset,config,device='cuda:0',cache_only=False,refresh=False,encoder_factory=None,
                 image_devices=None,image_batch_size=8):
        self.dataset=dataset;self.root=dataset.root;self.config=config;self.device=device
        self.cache_only=cache_only;self.refresh=refresh;self.encoder_factory=encoder_factory
        if image_batch_size < 1:
            raise ValueError('Image batch size must be positive')
        self.image_devices=image_devices;self.image_batch_size=image_batch_size
        self.encoders={};self.banks={};self.ocr_banks={}
        self.cost=dict(ocr_inference_seconds=0.,ocr_process_seconds=0.,encoding_seconds=0.,
                       model_loading_seconds=0.,cache_validation_seconds=0.,generated={},hits={})

    def ocr(self,population):
        if population in self.ocr_banks:
            return self.ocr_banks[population]
        path=self.root/f'cache/ocr/{population}.json'
        bank=read_json(path) if path.exists() else {}
        metadata_path=self.root/'cache/ocr/metadata.json'
        meta=read_json(metadata_path) if metadata_path.exists() else {}
        expected=read_json(self.root/'configs/ppocrv6.json')
        model_spec=read_json(self.root/'configs/ppocrv6_models.json')
        if meta.get('config')!=expected or meta.get('models')!=model_spec:
            bank={}
        rows=self.dataset.records(population);key=self.dataset.id_key(population)
        missing=[r[key] for r in rows if not (r[key] in bank and
            bank[r[key]].get('image_sha256')==r['sha256'] and
            isinstance(bank[r[key]].get('text'),str))]
        self.dataset.verify_images(population)
        if missing:
            if self.cache_only:
                raise ValueError(f'Missing/stale OCR ({len(missing)}): {missing[:20]}')
            start=time.monotonic()
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([python_executable(self.root, 'ocr', self.config.get('ocr_python')),'-B','-m','evaluation.ocr','--root',str(self.root),
                '--population',population,'--device',self.device.replace('cuda','gpu')],
                cwd=Path(__file__).resolve().parents[1],env=env,check=True)
            self.cost['ocr_process_seconds']+=time.monotonic()-start
            bank=read_json(path)
            runs=read_json(metadata_path).get('incremental_runs',[])
            if runs:
                self.cost['ocr_inference_seconds']+=runs[-1]['inference_seconds']
        self.cost['generated'][f'ocr/{population}']=len(missing)
        self.cost['hits'][f'ocr/{population}']=len(rows)-len(missing)
        self.ocr_banks[population]=bank
        return bank

    def encoder(self,model):
        if model not in self.encoders:
            factory=self.encoder_factory
            if factory is None:
                if model in IMAGE_ENCODERS:
                    from .dense import DenseEncoder, ParallelDenseEncoder
                    from functools import partial
                    factory=partial(ParallelDenseEncoder,devices=self.image_devices) if self.image_devices else DenseEncoder
                else:
                    from .encoders import Encoder
                    factory=Encoder
            started=time.monotonic()
            self.encoders[model]=factory(model,self.config['models'][model],self.root,self.device)
            self.cost['model_loading_seconds']+=time.monotonic()-started
        return self.encoders[model]

    def dense_images(self, model, population, rows, positions, contract):
        """Checkpoint small global-vector banks so interrupted encoding can resume."""
        path=self.root/f'cache/image_embeddings/{model}/{population}.partial.pt'
        signature=digest(dict(model=self.config['models'][model]['image_model'],contract=contract,
                              rows=[(r[self.dataset.id_key(population)],r['sha256']) for r in rows]))
        partial=torch.load(path,map_location='cpu',weights_only=True) if path.exists() and not self.refresh else {}
        values=partial.get('embeddings',{}) if partial.get('signature')==signature else {}
        values={i:v for i,v in values.items() if i in positions and valid_embedding(v,model,'image')}
        pending=[i for i in positions if i not in values]
        checkpoint_count=0
        for offset in range(0,len(pending),self.image_batch_size):
            indices=pending[offset:offset+self.image_batch_size]
            encoder=self.encoder(model)
            started=time.monotonic()
            encoded=encoder.encode_images([self.root/'dataset'/rows[i]['path'] for i in indices])
            self.cost['encoding_seconds']+=time.monotonic()-started
            for i,value in zip(indices,encoded,strict=True):
                if not valid_embedding(value,model,'image'):
                    raise ValueError(f'Invalid dense embedding at row {i}')
                values[i]=value
            checkpoint_count+=len(indices)
            if checkpoint_count>=128 or offset+len(indices)==len(pending):
                save_tensor(path,dict(signature=signature,embeddings=values))
                checkpoint_count=0
                print(f'Encoded {model}/image/{population}: {len(values)}/{len(positions)}',flush=True)
        return values

    def embeddings(self,model,view,population):
        key=(model,view,population)
        if key in self.banks:
            return self.banks[key]
        started=time.monotonic()
        spec=self.config['models'][model]
        if view=='text' and 'text_model' not in spec:
            raise ValueError(f'{model} only supports image embeddings')
        folder='image_embeddings' if view=='image' else 'ocr_embeddings'
        directory=self.root/'cache'/folder
        path=directory/model/f'{population}.pt'
        old=torch.load(path,map_location='cpu',weights_only=True,mmap=True) if path.exists() else {}
        rows=self.dataset.records(population);id_key=self.dataset.id_key(population)
        self.dataset.verify_images(population)
        ocr=self.ocr(population) if view=='text' else None
        expected_model=spec['image_model' if view=='image' else 'text_model']
        contract=spec['image_contract' if view=='image' else 'text_contract']
        compatible=(not self.refresh and old.get('model')==expected_model and old.get('view')==view
                    and old.get('schema')==f'release.{folder}.v1' and old.get('population')==population)
        metadata_path=directory/'metadata.json'
        metadata=read_json(metadata_path) if metadata_path.exists() else {}
        if view=='image' and 'contract' not in old:
            runtimes=metadata.get('models',{}).get(model,{}).get('runtimes',[])
            compatible=compatible and bool(runtimes) and all(r.get('contract')==contract for r in runtimes)
        else:
            compatible=compatible and old.get('contract')==contract
        index={identifier:i for i,identifier in enumerate(old.get('ids',[]))} if compatible else {}
        if len(index)!=len(old.get('ids',[])) and compatible:
            raise ValueError(f'Duplicate cache IDs: {path}')
        embeddings=[];provenance=[];counts=[];missing=[];has_text=[];text_hashes=[]
        for position,row in enumerate(rows):
            identifier=row[id_key];i=index.get(identifier)
            present=bool(ocr[identifier]['text'].strip()) if ocr is not None else True
            th=text_hash(ocr[identifier]['text']) if ocr is not None else None
            ok=i is not None
            if ok:
                for field,expected in [('sha256',row['sha256']),('source_sha256',row.get('source_sha256',row['sha256']))]:
                    ok=ok and len(old.get(field,[]))>i and old[field][i]==expected
                if view=='text':
                    ok=ok and all(len(old.get(f,[]))>i for f in ('text_sha256','has_text','input_token_counts'))
                    ok=ok and old['text_sha256'][i]==th and old['has_text'][i]==present
                    ok=ok and 0<=old['input_token_counts'][i]<=contract['context_limit']
                ok=ok and len(old.get('embeddings',[]))>i and len(old.get('provenance',[]))>i
                ok=ok and valid_embedding(old['embeddings'][i],model,view,present,
                                         old['input_token_counts'][i] if view=='text' else None)
            embeddings.append(old['embeddings'][i] if ok else None)
            provenance.append(old['provenance'][i] if ok else None)
            counts.append(old['input_token_counts'][i] if ok and view=='text' else 0)
            if not ok:
                missing.append(position)
            has_text.append(present);text_hashes.append(th)
        self.cost['cache_validation_seconds']+=time.monotonic()-started
        if missing and self.cache_only:
            raise ValueError(f'Missing/stale {key} ({len(missing)}): {[rows[i][id_key] for i in missing[:20]]}')
        # Check bytes whenever a complete stored bank is reused.
        recorded=metadata.get('models',{}).get(model,{}).get('files',{}).get(population,{}).get('sha256')
        if not missing and recorded and file_hash(path)!=recorded:
            raise ValueError(f'Cache file checksum differs: {path}')
        dense=self.dense_images(model,population,rows,missing,contract) if missing and model in IMAGE_ENCODERS else {}
        partial_directory=path.parent/(population+'.partial')
        partial_recipe=digest(dict(model=expected_model,contract=contract,view=view))
        for position in missing:
            row=rows[position];identifier=row[id_key]
            partial_path=partial_directory/(identifier+'.pt')
            partial=torch.load(partial_path,map_location='cpu',weights_only=True) if partial_path.exists() and not self.refresh else {}
            resumed=(partial.get('recipe')==partial_recipe and partial.get('image_sha256')==row['sha256']
                and partial.get('source_sha256')==row.get('source_sha256',row['sha256'])
                and partial.get('text_sha256')==text_hashes[position]
                and valid_embedding(partial.get('embedding'),model,view,has_text[position],partial.get('tokens')))
            if resumed:
                value=partial['embedding'];count=partial['tokens']
            elif position in dense:
                value=dense[position];count=0
            elif view=='text' and not has_text[position]:
                value=(torch.zeros(2048,dtype=torch.float32) if model=='Qwen3-VL-Embedding-2B'
                       else torch.empty(0,128,dtype=torch.float16));count=0
            else:
                encoder=self.encoder(model)
                start=time.monotonic()
                if view=='image':
                    value=encoder.encode_image(self.root/'dataset'/row['path']);count=0
                else:
                    value,count=encoder.encode_text(ocr[identifier]['text'])
                self.cost['encoding_seconds']+=time.monotonic()-start
            if not valid_embedding(value,model,view,has_text[position],count):
                raise ValueError(f'Invalid new embedding: {identifier}')
            if model not in IMAGE_ENCODERS and not resumed:
                save_tensor(partial_path,dict(recipe=partial_recipe,image_sha256=row['sha256'],
                    source_sha256=row.get('source_sha256',row['sha256']),text_sha256=text_hashes[position],
                    embedding=value,tokens=count))
            embeddings[position]=value;counts[position]=count
            if view=='image':
                provenance[position]=dict(kind='generated',contract=dict(contract,image_sha256=row['sha256']))
            else:
                provenance[position]=dict(kind='generated_text' if has_text[position] else 'empty_text',
                                          ocr_id=identifier,text_sha256=text_hashes[position])
            if model not in IMAGE_ENCODERS:
                print(f'Encoded {model}/{view}/{population}: {identifier}',flush=True)
        result=dict(schema=f'release.{folder}.v1',model=expected_model,family=spec['family'],view=view,
            population=population,ids=[r[id_key] for r in rows],paths=['dataset/'+r['path'] for r in rows],
            sha256=[r['sha256'] for r in rows],source_sha256=[r.get('source_sha256',r['sha256']) for r in rows],
            identities=[r['identity'] for r in rows],manifest=f'manifests/{population}.csv',
            manifest_sha256=file_hash(self.dataset.lists/f'{population}.csv'),
            embeddings=torch.stack(embeddings) if model in DENSE_DIMENSIONS else embeddings,
            provenance=provenance,contract=contract)
        if view=='text':
            result.update(ocr_source=f'cache/ocr/{population}.json',ocr_source_sha256=file_hash(
                self.root/f'cache/ocr/{population}.json'),text_field='text',text_sha256=text_hashes,
                has_text=has_text,input_token_counts=counts)
        # Preserve encoding provenance across label-only or path-only revisions.
        # The returned view uses current paths; the stored bank keeps its input manifest.
        fields=('ids','sha256','source_sha256')
        rewrite=bool(missing) or any(old.get(k)!=result[k] for k in fields)
        rewrite=rewrite or (view=='text' and old.get('ocr_source_sha256')!=result['ocr_source_sha256'])
        if rewrite:
            result['manifest']=f'manifests/{result["manifest_sha256"]}/{population}.csv'
            (directory/result['manifest']).parent.mkdir(parents=True,exist_ok=True)
            save_tensor(path,result)
            shutil.copyfile(self.dataset.lists/f'{population}.csv',directory/result['manifest'])
            self.update_metadata(model,view,population,result,path)
        if model in IMAGE_ENCODERS:
            path.with_name(f'{population}.partial.pt').unlink(missing_ok=True)
        elif partial_directory.exists():
            for row in rows:
                (partial_directory/(row[id_key]+'.pt')).unlink(missing_ok=True)
            if not any(partial_directory.iterdir()):
                partial_directory.rmdir()
        self.cost['generated']['/'.join(key)]=len(missing)
        self.cost['hits']['/'.join(key)]=len(rows)-len(missing)
        result['_file_sha256']=file_hash(path)
        self.banks[key]=result
        return result

    def update_metadata(self,model,view,population,bank,path):
        directory=path.parent.parent
        with (directory/'.metadata.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            mp=directory/'metadata.json';meta=read_json(mp) if mp.exists() else {}
            models=meta.setdefault('models',{});entry=models.setdefault(model,{})
            entry.update(model=bank['model'],family=bank['family'],directory=model,view=view)
            entry.setdefault('files',{})[population]=dict(path=f'{model}/{population}.pt',
                bytes=path.stat().st_size,sha256=file_hash(path))
            if view=='image':
                entry[population]=len(bank['ids'])
                entry['reused_'+population]=sum(p['kind']=='reused' for p in bank['provenance'])
                entry['generated_'+population]=sum(p['kind']=='generated' for p in bank['provenance'])
                if not entry.get('runtimes'):
                    source='dense.py' if model in IMAGE_ENCODERS else 'encoders.py'
                    entry['runtimes']=[dict(contract=bank['contract'],encoder_sha256=file_hash(Path(__file__).with_name(source)))]
            else:
                entry['contract']=bank['contract']
                entry.setdefault('counts',{})[population]=dict(records=len(bank['ids']),
                    nonempty_embeddings=sum(bank['has_text']),empty_ids=[i for i,h in zip(bank['ids'],bank['has_text']) if not h],
                    max_input_tokens=max(bank['input_token_counts'],default=0))
                runtime=entry.setdefault('runtime',{})
                runtime.setdefault('script_sha256',file_hash(Path(__file__).with_name('encoders.py')))
                # Each immutable bank can have a different historical CSV.
                # A current inventory mirror cannot replace that provenance.
                runtime['input_files']={}
                for pop in ('photos','references'):
                    input_path=directory/model/f'{pop}.pt'
                    if input_path.exists():
                        input_bank=bank if pop==population else torch.load(
                            input_path,map_location='cpu',weights_only=True,mmap=True)
                        relative=(directory/input_bank['manifest']).relative_to(self.root).as_posix()
                        runtime['input_files'][relative]=input_bank['manifest_sha256']
                    ocr_path=self.root/f'cache/ocr/{pop}.json'
                    if ocr_path.exists():
                        runtime['input_files'][ocr_path.relative_to(self.root).as_posix()]=file_hash(ocr_path)
            expected={name for name,spec in self.config['models'].items() if view=='image' or 'text_model' in spec}
            complete=expected<=models.keys() and all(
                set(models[name].get('files',{}))=={'photos','references'} and all(
                    (models[name].get(population) if view=='image' else
                     models[name].get('counts',{}).get(population,{}).get('records'))==len(self.dataset.records(population))
                    for population in ('photos','references')) for name in expected)
            meta['status']='complete' if complete and not meta.get('migration_in_progress') else 'building'
            if view=='image':
                meta.update(view='image',photos_per_model=len(self.dataset.photos),references_per_model=len(self.dataset.references),
                            total_embeddings=sum(m.get(p,0) for m in models.values() for p in ('photos','references')))
            else:
                meta.update(photos_per_model=len(self.dataset.photos),references_per_model=len(self.dataset.references))
                meta['total_records']=sum(c['records'] for m in models.values() for c in m.get('counts',{}).values())
                meta['total_nonempty_embeddings']=sum(c['nonempty_embeddings'] for m in models.values() for c in m.get('counts',{}).values())
            write_json(mp,meta)

    def release(self):
        for encoder in self.encoders.values():
            encoder.release()
        self.encoders.clear()
