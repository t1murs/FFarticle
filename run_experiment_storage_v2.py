"""Storage-version2 copy of the audited Stage 3B experiment runner.

Only physical compact-round encoding, format provenance and its storage review
gate differ. Equations, keys, conditions, N, phase and restoration calculations
remain the original audited code. Original run_experiment.py stays immutable.

Gated, checkpointed Stage 3B variance/production execution.

No research runs occur on import. The CLI refuses execution unless the recorded
scientific, implementation and design gates permit the requested stage.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from family_ai_fast import engine
from experiment_outputs import (OUTCOME_RECORDS,OUTCOME_KEYS,phase_vectors,phase_table,
    compact_round_table,restoration_table,summary_tables,parquet_writer,write_small_table)
from storage_v2 import packed_compact_round_table, packed_parquet_writer

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,path)
def log(path,action,**kw):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps({'at_utc':now(),'action':action,**kw},ensure_ascii=False)+'\n')

def verify_gates(args,manifest):
    science=load(args.science_gate)
    if science.get('scientific_gate_passed') is not True:raise RuntimeError('Scientific gate is not passed')
    implementation=load(args.engine_gate)
    if implementation.get('status')!='PASS':raise RuntimeError('Compiled engine has no independent PASS')
    outputs=load(args.outputs_gate)
    if outputs.get('status')!='PASS':raise RuntimeError('Output pipeline has no independent PASS')
    for rel,expected in outputs['code_hashes'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError(f'Audited output pipeline changed: {rel}')
    storage=load(args.storage_gate)
    if storage.get('status')!='PASS':raise RuntimeError('Storage version2 has no independent PASS')
    for rel,expected in storage['code_hashes'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError(f'Audited storage version2 changed: {rel}')
    frozen=load(ROOT/'audit/engine_source_freeze.json')
    for rel,expected in frozen['code_hashes'].items():
        if sha(ROOT/rel)!=expected:raise RuntimeError(f'Audited engine changed: {rel}')
    design=load(args.design_gate)
    if design.get('status')!='PASS' or design.get('design_frozen') is not True:
        raise RuntimeError('Experiment design is not frozen and verified')
    if design.get('manifest_sha256')!=sha(args.manifest):raise RuntimeError('Manifest hash differs from design freeze')
    if design.get('variance_authorized') is not True:raise RuntimeError('Variance stage is not authorized')
    paths=[args.science_gate,args.engine_gate,args.outputs_gate,args.storage_gate,args.design_gate,ROOT/'audit/engine_source_freeze.json']
    if args.stage=='variance':N=128;namespace=12
    else:
        precision=load(args.precision_freeze)
        if precision.get('production_authorized') is not True:raise RuntimeError('Production precision is not frozen/authorized')
        N=precision['selected_N'];namespace=21
        if isinstance(N,bool) or not isinstance(N,int) or not 256<=N<=8192 or N%64:
            raise RuntimeError('Selected N violates frozen E17 constraints')
        if precision.get('manifest_sha256',sha(args.manifest))!=sha(args.manifest):
            raise RuntimeError('Precision freeze refers to another experimental design')
        paths.append(args.precision_freeze)
    base_cells=manifest['base_cells']
    if len(base_cells)!=876:raise RuntimeError('The complete frozen design requires876basecells')
    if len({int(c['base_index']) for c in base_cells})!=876:raise RuntimeError('Duplicate base indices')
    for cell in base_cells:
        if cell['parameters']['J']!=20 or cell['parameters']['T']!=60:
            raise RuntimeError('This runner requires the frozen20domain60rounddesign')
    return N,namespace,{str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else str(p):sha(p) for p in paths}

def verify_cell(path,manifest_hash,run_signature):
    record=load(path/'cell_manifest.json')
    if record.get('status')!='PASS' or record['experiment_manifest_sha256']!=manifest_hash:
        raise RuntimeError(f'Invalid completed-cell checkpoint:{path}')
    if record.get('run_signature')!=run_signature:raise RuntimeError(f'Cell belongs to a different run:{path}')
    for row in record['files']:
        if sha(path/row['name'])!=row['sha256']:raise RuntimeError(f'Checkpoint payload hash failed:{path/row["name"]}')
    return record

def execute_cell(cell,N,namespace,args,manifest_hash,run_signature,save_rounds,save_restoration):
    temp=args.output/'.inprogress'/f'cell_{int(cell["base_index"]):04d}_{uuid.uuid4().hex[:10]}'
    temp.mkdir(parents=True)
    conditions=cell['unique_conditions'];C=len(conditions);p=cell['parameters']
    ids=[str(x.get('physical_id',i)) for i,x in enumerate(conditions)]
    vectors=np.empty((C,N,20),dtype=np.float64)
    phase_writer=round_writer=rest_writer=None
    fallback={k:0 for k in ['C_pre','Q','H_post','R']}
    totals={'rng_seconds':0.0,'transition_seconds':0.0,'serialization_seconds':0.0}
    start=time.perf_counter();batch_records=[]
    try:
        for first in range(1,N+1,args.batch_size):
            firm_ids=np.arange(first,min(N+1,first+args.batch_size),dtype=np.int64)
            t=time.perf_counter()
            ex=engine.generate_exogenous(p,len(firm_ids),int(cell['parent_cell_id']),namespace,
                                         int(cell['design_id']),0,firm_ids)
            rngtime=time.perf_counter()-t;totals['rng_seconds']+=rngtime
            t=time.perf_counter();run=engine.simulate_batch(p,ex,conditions,debug=False)
            transitiontime=time.perf_counter()-t;totals['transition_seconds']+=transitiontime
            scores=run['firm_round'];t=time.perf_counter()
            phase=phase_vectors(scores);vectors[:,firm_ids-1,:]=phase
            pt=phase_table(phase,ids,firm_ids,cell['base_index'])
            if phase_writer is None:phase_writer=parquet_writer(temp/'phase_vectors.parquet',pt.schema)
            phase_writer.write_table(pt,row_group_size=len(pt))
            if save_rounds:
                rt,counts=packed_compact_round_table(scores,ids,firm_ids,cell['base_index'])
                for k in fallback:fallback[k]+=counts[k]
                if round_writer is None:round_writer=packed_parquet_writer(temp/'compact_rounds.parquet',rt.schema)
                round_writer.write_table(rt,row_group_size=len(rt))
            if save_restoration:
                rest=restoration_table(scores,p,conditions,ex,ids,firm_ids,cell['base_index'])
                if rest_writer is None:rest_writer=parquet_writer(temp/'restoration.parquet',rest.schema)
                rest_writer.write_table(rest,row_group_size=len(rest))
            serialtime=time.perf_counter()-t;totals['serialization_seconds']+=serialtime
            batch_records.append({'first_firm_id':int(firm_ids[0]),'last_firm_id':int(firm_ids[-1]),
                                  'rng_seconds':rngtime,'transition_seconds':transitiontime,
                                  'serialization_seconds':serialtime})
        for writer in [phase_writer,round_writer,rest_writer]:
            if writer is not None:writer.close()
        phase_writer=round_writer=rest_writer=None
        outcome_table,contrast_table=summary_tables(vectors,cell)
        write_small_table(temp/'condition_summary.parquet',outcome_table)
        write_small_table(temp/'contrast_summary.parquet',contrast_table)
        primary=contrast_table.filter(contrast_table['precision_primary'])
        if len(primary):
            variance=primary['variance'].to_numpy();at=int(np.argmax(variance));row=primary.slice(at,1).to_pylist()[0]
            max_variance=float(variance[at]);max_case={k:row[k] for k in ['contrast_id','key','variance','N']}
        else:max_variance=0.;max_case=None
        inventory=[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(temp.glob('*.parquet'))]
        record={'status':'PASS','completed_at_utc':now(),'stage':args.stage,'namespace':namespace,
                'base_index':cell['base_index'],'family':cell['family'],'cell_key':cell['cell_key'],
                'parent_cell_id':str(cell['parent_cell_id']),'design_id':cell['design_id'],
                'N':N,'logical_conditions':len(cell['conditions']),'physical_conditions':C,
                'condition_aliases':{str(x['condition_id']):str(x['physical_id']) for x in cell['conditions']},
                'experiment_manifest_sha256':manifest_hash,'run_signature':run_signature,
                'elapsed_seconds':time.perf_counter()-start,'timings':totals,'batches':batch_records,
                'data_origin':'SIMULATED','phase_rows':C*N,'phase_outcomes':20,
                'compact_rounds_saved':save_rounds,'restoration_saved':save_restoration,
                'storage_format_version':2,'storage_compression':'Zstd9; packed182float64;64MiBtargetpage',
                'storage_probe_only':args.stage=='variance' and save_rounds,
                'compact_identity_fallback_rows':fallback,'maximum_primary_paired_variance':max_variance,
                'maximum_variance_case':max_case,'files':inventory}
        atomic_json(temp/'cell_manifest.json',record)
        final=args.output/'cells'/f'cell_{int(cell["base_index"]):04d}'
        if final.exists():raise RuntimeError(f'Refusing to overwrite existing checkpoint:{final}')
        os.replace(temp,final)
        return record
    except Exception:
        for writer in [phase_writer,round_writer,rest_writer]:
            if writer is not None:writer.close()
        atomic_json(temp/'failed_attempt.json',{'at_utc':now(),'stage':args.stage,'base_index':cell['base_index'],
                    'status':'FAILED','traceback':traceback.format_exc(),'completed_batches':batch_records})
        raise

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['variance','production'],required=True)
    ap.add_argument('--manifest',type=Path,default=ROOT/'config/experiment_manifest.json')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--science-gate',type=Path,default=ROOT/'audit/science_preproduction_gate.json')
    ap.add_argument('--engine-gate',type=Path,default=ROOT/'audit/fast_equivalence_review.json')
    ap.add_argument('--outputs-gate',type=Path,default=ROOT/'audit/output_pipeline_review.json')
    ap.add_argument('--storage-gate',type=Path,default=ROOT/'audit/storage_v2_review.json')
    ap.add_argument('--design-gate',type=Path,default=ROOT/'audit/design_freeze.json')
    ap.add_argument('--precision-freeze',type=Path,default=ROOT/'config/precision_freeze.json')
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    for name in ['manifest','science_gate','engine_gate','outputs_gate','storage_gate','design_gate','precision_freeze']:
        setattr(args,name,getattr(args,name).resolve())
    args.output=(args.output or ROOT/'data'/f'{args.stage}_v2').resolve()
    if not 1<=args.batch_size<=8192:raise ValueError('Invalid numerical batch size')
    if not 1<=args.threads<=256:raise ValueError('Invalid numerical thread count')
    manifest=load(args.manifest);N,namespace,gates=verify_gates(args,manifest)
    manifest_hash=sha(args.manifest)
    code_files=[Path(__file__).resolve(),ROOT/'src/storage_v2.py',ROOT/'src/experiment_outputs.py',ROOT/'src/family_ai_fast/engine.py',
                ROOT/'src/family_ai_fast/engine.cpp',ROOT/'src/family_ai_fast/_backend.so']
    code_hashes={str(p.relative_to(ROOT)):sha(p) for p in code_files}
    signature={'storage_format_version':2,'stage':args.stage,'namespace':namespace,'N':N,'manifest_sha256':manifest_hash,
               'code_hashes':code_hashes,'gates':gates}
    run_signature=hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'cells').mkdir(exist_ok=True);(args.output/'.inprogress').mkdir(exist_ok=True)
    run_path=args.output/'run_manifest.json';events=args.output/'run_events.jsonl'
    if run_path.exists():
        if not args.resume:raise RuntimeError('Output already has a run manifest; use --resume or a new directory')
        previous=load(run_path)
        if previous.get('run_signature')!=run_signature:raise RuntimeError('Cannot resume with changed code, gates, design or N')
    engine.set_threads(args.threads)
    probes=set(manifest.get('storage_probe_base_indices',[1,109,365,621]))
    cells=manifest['base_cells'];completed=[];start=time.perf_counter()
    run={'started_at_utc':now(),'status':'RUNNING','run_signature':run_signature,**signature,
         'base_cells_expected':len(cells),'paired_histories_per_base_cell':N,'data_origin':'SIMULATED',
         'batch_size':args.batch_size,'threads':args.threads,'primary_outcomes':OUTCOME_RECORDS,
         'storage_probe_base_indices':sorted(probes),'round_storage':'Packed182float64 Parquet version2; all identity exceptions retained',
         'completed_cells':[],'uncompleted_base_indices':[x['base_index'] for x in cells]}
    atomic_json(run_path,run);log(events,'RUN_START_OR_RESUME',stage=args.stage,N=N,namespace=namespace,run_signature=run_signature)
    try:
        for cell in cells:
            final=args.output/'cells'/f'cell_{int(cell["base_index"]):04d}'
            if final.exists():record=verify_cell(final,manifest_hash,run_signature);action='CELL_REUSED_VERIFIED'
            else:
                record=execute_cell(cell,N,namespace,args,manifest_hash,run_signature,
                                     args.stage=='production' or cell['base_index'] in probes,
                                     args.stage=='production');action='CELL_COMPLETED'
            completed.append(record)
            concise={k:record[k] for k in ['base_index','family','physical_conditions','elapsed_seconds','maximum_primary_paired_variance']}
            concise['bytes']=sum(x['bytes'] for x in record['files'])
            run['completed_cells'].append(concise)
            run['uncompleted_base_indices']=[x['base_index'] for x in cells[len(completed):]]
            run['elapsed_current_attempt_seconds']=time.perf_counter()-start
            atomic_json(run_path,run);log(events,action,**concise)
            print(json.dumps({'stage':args.stage,'completed':len(completed),'total':len(cells),**concise}),flush=True)
        maximum=max(completed,key=lambda x:x['maximum_primary_paired_variance'])
        run.update({'status':'PASS','completed_at_utc':now(),'underlying_histories':len(cells)*N,
                    'logical_conditional_trajectories':sum(len(x['conditions'])*N for x in cells),
                    'physical_conditional_trajectories':sum(len(x['unique_conditions'])*N for x in cells),
                    'maximum_primary_paired_variance':maximum['maximum_primary_paired_variance'],
                    'maximum_variance_case':{'base_index':maximum['base_index'],**(maximum['maximum_variance_case'] or {})},
                    'total_payload_bytes':sum(sum(y['bytes'] for y in x['files']) for x in completed),
                    'uncompleted_base_indices':[]})
        atomic_json(run_path,run);log(events,'RUN_COMPLETED',status='PASS',payload_bytes=run['total_payload_bytes'])
        print(json.dumps({'status':'PASS','stage':args.stage,'N':N,'cells':len(completed),
                          'maximum_primary_paired_variance':run['maximum_primary_paired_variance'],
                          'payload_bytes':run['total_payload_bytes']}),flush=True)
    except Exception:
        run.update({'status':'FAILED','failed_at_utc':now(),'traceback':traceback.format_exc()})
        atomic_json(run_path,run);log(events,'RUN_FAILED',traceback=traceback.format_exc());raise

if __name__=='__main__':main()
