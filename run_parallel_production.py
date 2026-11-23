"""Thin process coordinator around the immutable storage-version2 runner.

Only independent base cells are scheduled in parallel. The frozen execute_cell
function controls every RNG key, transition, phase estimate, output and hash.
The parent alone updates the run-level checkpoint in sorted base-index order.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,multiprocessing,os,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from types import SimpleNamespace
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_experiment_storage_v2 as frozen
from family_ai_fast import engine
from experiment_outputs import OUTCOME_RECORDS

def worker_initialize(threads):
    engine.set_threads(threads)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

def worker_execute(cell,N,namespace,settings,manifest_hash,run_signature):
    args=SimpleNamespace(output=Path(settings['output']),stage='production',batch_size=settings['batch_size'])
    return frozen.execute_cell(cell,N,namespace,args,manifest_hash,run_signature,True,True)

def drive_cells(executor,cells,worker,worker_args,on_result,on_error,max_inflight):
    """Bounded scheduling; stop submissions after failure, drain live jobs.

    Executor and worker are injectable for deterministic scheduling tests. This
    helper contains no simulation or file mutation. Completion callbacks remain
    in the parent, never in worker processes.
    """
    remaining=iter(cells);active={};exhausted=False;errors=[]
    while True:
        while not errors and not exhausted and len(active)<max_inflight:
            try:cell=next(remaining)
            except StopIteration:exhausted=True;break
            active[executor.submit(worker,cell,*worker_args)]=cell
        if not active:break
        done,_=wait(active,return_when=FIRST_COMPLETED)
        # Sort simultaneous completions for stable parent event recording.
        for future in sorted(done,key=lambda f:active[f]['base_index']):
            cell=active.pop(future)
            try:record=future.result()
            except Exception as error:
                failure={'base_index':cell['base_index'],'exception':repr(error),'traceback':traceback.format_exc()}
                errors.append(failure);on_error(cell,failure)
            else:on_result(cell,record)
    return errors

def validate_coordinator_gate(path):
    gate=frozen.load(path)
    if gate.get('status')!='PASS':raise RuntimeError('Parallel coordinator has no independent PASS')
    for rel,expected in gate['code_hashes'].items():
        if frozen.sha(ROOT/rel)!=expected:raise RuntimeError(f'Audited coordinator changed:{rel}')
    return gate

def concise(record):
    row={key:record[key] for key in ['base_index','family','physical_conditions','elapsed_seconds','maximum_primary_paired_variance']}
    row['bytes']=sum(x['bytes'] for x in record['files'])
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=ROOT/'config/experiment_manifest.json')
    ap.add_argument('--output',type=Path,default=ROOT/'data/production_v2')
    ap.add_argument('--science-gate',type=Path,default=ROOT/'audit/science_preproduction_gate.json')
    ap.add_argument('--engine-gate',type=Path,default=ROOT/'audit/fast_equivalence_review.json')
    ap.add_argument('--outputs-gate',type=Path,default=ROOT/'audit/output_pipeline_review.json')
    ap.add_argument('--storage-gate',type=Path,default=ROOT/'audit/storage_v2_review.json')
    ap.add_argument('--design-gate',type=Path,default=ROOT/'audit/design_freeze.json')
    ap.add_argument('--precision-freeze',type=Path,default=ROOT/'config/precision_freeze.json')
    ap.add_argument('--coordinator-gate',type=Path,default=ROOT/'audit/parallel_coordinator_review.json')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--threads-per-worker',type=int,default=2)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();args.stage='production'
    for name in ['manifest','output','science_gate','engine_gate','outputs_gate','storage_gate','design_gate','precision_freeze','coordinator_gate']:
        setattr(args,name,getattr(args,name).resolve())
    if not 1<=args.workers<=64 or not 1<=args.threads_per_worker<=256 or not 1<=args.batch_size<=8192:
        raise ValueError('Invalid technical process/thread/batch setting')
    manifest=frozen.load(args.manifest);N,namespace,gates=frozen.verify_gates(args,manifest)
    validate_coordinator_gate(args.coordinator_gate)
    gates[str(args.coordinator_gate.relative_to(ROOT)) if args.coordinator_gate.is_relative_to(ROOT) else str(args.coordinator_gate)]=frozen.sha(args.coordinator_gate)
    manifest_hash=frozen.sha(args.manifest)
    code_files=[Path(__file__).resolve(),ROOT/'scripts/run_experiment_storage_v2.py',ROOT/'src/storage_v2.py',
                ROOT/'src/experiment_outputs.py',ROOT/'src/family_ai_fast/engine.py',
                ROOT/'src/family_ai_fast/engine.cpp',ROOT/'src/family_ai_fast/_backend.so']
    code_hashes={str(p.relative_to(ROOT)):frozen.sha(p) for p in code_files}
    signature={'execution_coordinator':'independent_base_cell_processes_v1','storage_format_version':2,
               'stage':'production','namespace':namespace,'N':N,'manifest_sha256':manifest_hash,
               'code_hashes':code_hashes,'gates':gates}
    run_signature=hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'coordinator.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another process coordinator is already writing this output directory')
    lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'at_utc':frozen.now(),'run_signature':run_signature}));lock.flush()
    (args.output/'cells').mkdir(exist_ok=True);(args.output/'.inprogress').mkdir(exist_ok=True)
    run_path=args.output/'run_manifest.json';events=args.output/'run_events.jsonl';previous=None
    if run_path.exists():
        if not args.resume:raise RuntimeError('Output already has a run manifest; use --resume or a documented new directory')
        previous=frozen.load(run_path)
        if previous.get('run_signature')!=run_signature:raise RuntimeError('Cannot resume with changed code, gates, design or N')
    cells=manifest['base_cells'];by_index={int(c['base_index']):c for c in cells};completed={};pending=[]
    for cell in cells:
        path=args.output/'cells'/f'cell_{int(cell["base_index"]):04d}'
        if path.exists():
            record=frozen.verify_cell(path,manifest_hash,run_signature)
            if int(record['base_index'])!=int(cell['base_index']):raise RuntimeError('Checkpoint base index mismatch')
            completed[int(cell['base_index'])]=record
        else:pending.append(cell)
    start=time.perf_counter()
    run={'started_at_utc':previous.get('started_at_utc',frozen.now()) if previous else frozen.now(),
         'current_attempt_started_at_utc':frozen.now(),'status':'RUNNING','run_signature':run_signature,**signature,
         'base_cells_expected':len(cells),'paired_histories_per_base_cell':N,'data_origin':'SIMULATED',
         'batch_size':args.batch_size,'workers':args.workers,'engine_threads_per_worker':args.threads_per_worker,
         'process_start_method':'spawn','primary_outcomes':OUTCOME_RECORDS,
         'round_storage':'Packed182float64 Parquet version2; all identity exceptions retained',
         'completed_cells':[],'uncompleted_base_indices':[],'worker_failures':[]}
    def checkpoint():
        run['completed_cells']=[concise(completed[i]) for i in sorted(completed)]
        run['uncompleted_base_indices']=sorted(set(by_index)-set(completed))
        run['elapsed_current_attempt_seconds']=time.perf_counter()-start
        frozen.atomic_json(run_path,run)
    def on_result(cell,record):
        i=int(cell['base_index'])
        if i in completed:raise RuntimeError('Duplicate worker completion')
        if record.get('status')!='PASS' or record.get('run_signature')!=run_signature or int(record['base_index'])!=i:
            raise RuntimeError('Worker returned an invalid completed checkpoint')
        completed[i]=record;checkpoint()
        row=concise(record);frozen.log(events,'CELL_COMPLETED',**row)
        print(json.dumps({'stage':'production','completed':len(completed),'total':len(cells),**row}),flush=True)
    def on_error(cell,failure):
        run['worker_failures'].append(failure);checkpoint();frozen.log(events,'WORKER_CELL_FAILED',**failure)
        print(json.dumps({'stage':'production','worker_failed_base_index':cell['base_index'],'exception':failure['exception']}),flush=True)
    checkpoint();frozen.log(events,'PARALLEL_RUN_START_OR_RESUME',N=N,namespace=namespace,workers=args.workers,
                           engine_threads_per_worker=args.threads_per_worker,resumed_cells=len(completed),run_signature=run_signature)
    settings={'output':str(args.output),'batch_size':args.batch_size}
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_initialize,initargs=(args.threads_per_worker,)) as executor:
            failures=drive_cells(executor,pending,worker_execute,(N,namespace,settings,manifest_hash,run_signature),
                                 on_result,on_error,args.workers)
        if failures:raise RuntimeError(f'{len(failures)} cell worker(s) failed; no new cells submitted after the first failure')
        if len(completed)!=len(cells):raise RuntimeError('The complete design was not executed')
        maximum=max(completed.values(),key=lambda x:x['maximum_primary_paired_variance'])
        run.update({'status':'PASS','completed_at_utc':frozen.now(),'underlying_histories':len(cells)*N,
                    'logical_conditional_trajectories':sum(len(c['conditions'])*N for c in cells),
                    'physical_conditional_trajectories':sum(len(c['unique_conditions'])*N for c in cells),
                    'maximum_primary_paired_variance':maximum['maximum_primary_paired_variance'],
                    'maximum_variance_case':{'base_index':maximum['base_index'],**(maximum['maximum_variance_case'] or {})},
                    'total_payload_bytes':sum(sum(f['bytes'] for f in c['files']) for c in completed.values())})
        checkpoint();frozen.log(events,'PARALLEL_RUN_COMPLETED',status='PASS',payload_bytes=run['total_payload_bytes'])
        print(json.dumps({'status':'PASS','stage':'production','N':N,'cells':len(completed),
                          'payload_bytes':run['total_payload_bytes']}),flush=True)
    except BaseException:
        # Running jobs have drained when leaving the executor context. A
        # completed on-disk cell that missed a callback remains resumable.
        run.update({'status':'FAILED','failed_at_utc':frozen.now(),'traceback':traceback.format_exc()})
        checkpoint();frozen.log(events,'PARALLEL_RUN_FAILED',traceback=traceback.format_exc());raise
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
