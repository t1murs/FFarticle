"""Aggregate a complete, verified Stage 3B production run without new simulation.

This script reads only saved checkpoint outputs. It does not change contrasts,
choose a model, test a hypothesis, or classify the substantive findings.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_outputs import OUTCOME_KEYS,RESTORATION_DEFINITIONS,RESTORATION_CATEGORIES
from storage_v2 import unpack_compact_table

RESTORATION_CONTRAST_FAMILIES={'core_mode','pure_access','principal_interaction'}
STAT_COLUMNS=[(prefix+key,pa.float64()) for key in OUTCOME_KEYS for prefix in ['', 'se__', 'var__']]
COMMON=[('base_index',pa.int32()),('base_family',pa.string()),('design_id',pa.int32()),
        ('cell_key',pa.string()),('parent_cell_id',pa.string())]
CONTRAST_SCHEMA=pa.schema(COMMON+[
    ('contrast_id',pa.string()),('contrast_family',pa.string()),('context',pa.string()),
    ('interface',pa.string()),('left',pa.string()),('right',pa.string()),('L',pa.int8()),
    ('learning_rule',pa.string()),('component_label',pa.string()),('metadata_json',pa.string()),
    ('terms_json',pa.string()),('precision_primary',pa.bool_()),('N',pa.int32()),
    ('data_origin',pa.string())]+STAT_COLUMNS)
CONDITION_SCHEMA=pa.schema(COMMON+[
    ('condition_id',pa.string()),('physical_id',pa.string()),('context',pa.string()),('interface',pa.string()),
    ('L',pa.int8()),('learning_rule',pa.string()),('variant_parameter_id',pa.string()),
    ('parameter_overrides_json',pa.string()),('r',pa.float64()),('c_s',pa.float64()),
    ('epsilon',pa.float64()),('v',pa.float64()),('resolved_chi',pa.float64()),
    ('resolved_gamma',pa.float64()),('resolved_lambda_q',pa.float64()),('resolved_d',pa.float64()),
    ('N',pa.int32()),('data_origin',pa.string())]+STAT_COLUMNS)
RESTORATION_SCHEMA=pa.schema(COMMON+[
    ('physical_id',pa.string()),('physical_index',pa.int32()),('definition_index',pa.int8()),
    ('baseline_definition',pa.string()),('tolerance',pa.float64()),('N',pa.int32()),
    ('no_actual_shock_count',pa.int32()),('no_initial_material_loss_count',pa.int32()),
    ('restoration_confirmed_count',pa.int32()),('initial_loss_unrestored_count',pa.int32()),
    ('initial_material_loss_count',pa.int32()),('baseline_mean',pa.float64()),
    ('Q21_mean',pa.float64()),('Q40_mean',pa.float64()),('initial_loss_mean_full_sample',pa.float64()),
    ('mean_start_round_among_restored',pa.float64()),('mean_confirmation_round_among_restored',pa.float64()),
    ('mean_start_delay_from21_among_restored',pa.float64()),('mean_confirmation_delay_from21_among_restored',pa.float64()),
    ('Q40_improved_among_initial_loss_unrestored_count',pa.int32()),
    ('Q40_improved_among_any_unrestored_count',pa.int32()),('any_unrestored_count',pa.int32()),
    ('censored_after_initial_loss_count',pa.int32()),('censoring_round',pa.int16()),
    ('actual_shock_count',pa.int32()),('sustained_maintenance_count',pa.int32()),
    ('category1_later_below_benchmark_count',pa.int32()),
    ('sustained_maintenance_proportion_all',pa.float64()),('sustained_maintenance_proportion_affected',pa.float64()),
    ('category1_later_below_proportion_all',pa.float64()),('category1_later_below_proportion_affected',pa.float64()),
    ('initial_loss_threshold_roundoff_disagreement_count',pa.int32()),
    ('data_origin',pa.string())])
PAIRED_RESTORATION_SCHEMA=pa.schema(COMMON+[
    ('contrast_id',pa.string()),('contrast_family',pa.string()),('context',pa.string()),
    ('learning_rule',pa.string()),('L',pa.int8()),('definition_index',pa.int8()),
    ('baseline_definition',pa.string()),('tolerance',pa.float64()),('category',pa.int8()),
    ('category_label',pa.string()),('N',pa.int32()),('mean_probability_contrast',pa.float64()),
    ('variance',pa.float64()),('sd',pa.float64()),('mcse',pa.float64()),
    ('terms_json',pa.string()),('data_origin',pa.string())])

def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def jdump(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False))
def common(cell):
    return {'base_index':cell['base_index'],'base_family':cell['family'],'design_id':cell['design_id'],
            'cell_key':cell['cell_key'],'parent_cell_id':str(cell['parent_cell_id'])}
def column_array(table,name):
    col=table[name].combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        return col.values.to_numpy(zero_copy_only=False).reshape(len(col),col.type.list_size)
    return col.to_numpy(zero_copy_only=False)
def parquet_writer(path,schema):
    meta={b'stage3b_aggregation':jdump({'data_origin':'SIMULATED','phase_outcomes':OUTCOME_KEYS,
          'purpose':'Descriptive transformation of a complete production run; no new simulation',
          'statistics':'Full paired sample variance ddof1; MCSE sqrt(variance/N)'}).encode()}
    numeric=[f.name for f in schema if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)]
    dictionary=[f.name for f in schema if pa.types.is_string(f.type)]
    return pq.ParquetWriter(path,schema.with_metadata(meta),compression='zstd',compression_level=3,
                           use_dictionary=dictionary,use_byte_stream_split=numeric)

def verify_completed_run(run_dir,manifest_path,manifest):
    record=load(run_dir/'run_manifest.json')
    if record.get('status')!='PASS' or record.get('stage')!='production':
        raise RuntimeError('Aggregation requires a complete production PASS; no pilot/partial aggregation')
    if record.get('manifest_sha256')!=sha(manifest_path):raise RuntimeError('Production manifest hash mismatch')
    if len(manifest['base_cells'])!=876 or record.get('base_cells_expected')!=876:
        raise RuntimeError('Expected the complete876cell frozen design')
    if record.get('uncompleted_base_indices') or len(record.get('completed_cells',[]))!=876:
        raise RuntimeError('Run still contains incomplete base cells')
    expected={int(c['base_index']) for c in manifest['base_cells']}
    actual={int(c['base_index']) for c in record['completed_cells']}
    if actual!=expected:raise RuntimeError('Completed base-cell inventory differs from design')
    return record

def verify_input_cell(cell_dir,cell,run):
    record=load(cell_dir/'cell_manifest.json')
    if record.get('status')!='PASS' or record.get('stage')!='production':raise RuntimeError('Incomplete/nonproduction cell')
    for field in ['run_signature','N']:
        if record.get(field)!=run[field]:raise RuntimeError(f'Cell {field} differs from complete run')
    if record.get('experiment_manifest_sha256')!=run['manifest_sha256']:
        raise RuntimeError('Cell design identity mismatch')
    if int(record['base_index'])!=int(cell['base_index']):raise RuntimeError('Base index mismatch')
    if record['physical_conditions']!=len(cell['unique_conditions']):raise RuntimeError('Physical count mismatch')
    inventory={r['name']:r for r in record['files']}
    for name in ['condition_summary.parquet','contrast_summary.parquet','restoration.parquet','compact_rounds.parquet']:
        item=inventory[name];path=cell_dir/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise RuntimeError(f'Input checkpoint payload failed integrity check:{path}')
    return record

def stats_lookup(table,id_column,expected_ids,N):
    lookup={str(i):{} for i in expected_ids}
    if table.num_rows!=len(lookup)*20:raise RuntimeError('Unexpected summary row count')
    means=table['mean'].to_numpy();variances=table['variance'].to_numpy();ses=table['mcse'].to_numpy()
    if np.any(table['N'].to_numpy()!=N) or set(table['data_origin'].unique().to_pylist())!={'SIMULATED'}:
        raise RuntimeError('Summary provenance/N mismatch')
    if not (np.isfinite(means).all() and np.isfinite(variances).all() and np.isfinite(ses).all()) or np.any(variances<0) or np.any(ses<0):
        raise RuntimeError('Invalid finite Monte Carlo statistics')
    if not np.isclose(ses,np.sqrt(variances/N),rtol=1e-13,atol=1e-18).all():
        raise RuntimeError('Stored MCSE is inconsistent with paired variance/N')
    for identity,key,mean,se,variance in zip(table[id_column].to_pylist(),table['key'].to_pylist(),means,ses,variances):
        identity=str(identity)
        if identity not in lookup or key not in OUTCOME_KEYS or key in lookup[identity]:
            raise RuntimeError('Unknown/duplicate summary entity or outcome')
        lookup[identity][key]=(float(mean),float(se),float(variance))
    if any(len(x)!=20 for x in lookup.values()):raise RuntimeError('Incomplete summary outcomes')
    return lookup

def add_stats(row,stats):
    for key,(mean,se,var) in stats.items():row.update({key:mean,'se__'+key:se,'var__'+key:var})
    return row

def aggregate_primary(cell_dir,cell,N):
    required=['key','N','data_origin','mean','variance','mcse']
    physical=stats_lookup(pq.read_table(cell_dir/'condition_summary.parquet',columns=['physical_id']+required),'physical_id',
                          [x['physical_id'] for x in cell['unique_conditions']],N)
    comparisons=stats_lookup(pq.read_table(cell_dir/'contrast_summary.parquet',columns=['contrast_id']+required),'contrast_id',
                             [x['contrast_id'] for x in cell['contrasts']],N)
    contrast_rows=[];condition_rows=[]
    for contrast in cell['contrasts']:
        meta=contrast.get('metadata',{})
        row={**common(cell),'contrast_id':contrast['contrast_id'],'contrast_family':contrast['family'],
             **{k:meta.get(k) for k in ['context','interface','left','right','L','learning_rule','component_label']},
             'metadata_json':jdump(meta),'terms_json':jdump(contrast['terms']),
             'precision_primary':bool(contrast.get('precision_primary',False)),'N':N,'data_origin':'SIMULATED'}
        contrast_rows.append(add_stats(row,comparisons[str(contrast['contrast_id'])]))
    for condition in cell['conditions']:
        p={**cell['parameters'],**condition.get('parameters',{})}
        row={**common(cell),'condition_id':condition['condition_id'],'physical_id':str(condition['physical_id']),
             'context':condition['block'],'interface':condition['interface'],'L':condition['L'],
             'learning_rule':condition['learning_rule'],'variant_parameter_id':str(condition['variant_parameter_id']),
             'parameter_overrides_json':jdump(condition.get('parameters',{})),
             **{key:float(condition['mode'][key]) for key in ['r','c_s','epsilon','v']},
             **{'resolved_'+key:float(p[key]) for key in ['chi','gamma','lambda_q','d']},
             'N':N,'data_origin':'SIMULATED'}
        condition_rows.append(add_stats(row,physical[str(condition['physical_id'])]))
    return pa.Table.from_pylist(contrast_rows,schema=CONTRAST_SCHEMA),pa.Table.from_pylist(condition_rows,schema=CONDITION_SCHEMA)

def aggregate_restoration(path,cell,N,round_path):
    C=len(cell['unique_conditions']);counts=np.zeros((C,6,4),dtype=np.int64)
    baselines=np.zeros((C,2));initial_loss=np.zeros((C,2));q21=np.zeros(C);q40=np.zeros(C)
    start_sums=np.zeros((C,6),dtype=np.int64);confirm_sums=np.zeros_like(start_sums)
    improved_loss_unrestored=np.zeros((C,6),dtype=np.int64);improved_any_unrestored=np.zeros_like(improved_loss_unrestored)
    maintained=np.zeros((C,6),dtype=np.int64);delayed=np.zeros_like(maintained);threshold_disagreement=np.zeros_like(maintained)
    paired=[c for c in cell['contrasts'] if c['family'] in RESTORATION_CONTRAST_FAMILIES]
    sums=np.zeros((len(paired),6,4));squares=np.zeros_like(sums)
    physical_index={str(c['physical_id']):i for i,c in enumerate(cell['unique_conditions'])}
    logical={c['condition_id']:physical_index[str(c['physical_id'])] for c in cell['conditions']}
    coefficients=[]
    for contrast in paired:
        weights={}
        for term in contrast['terms']:
            i=logical[term['condition_id']];weights[i]=weights.get(i,0.)+float(term['weight'])
        coefficients.append({i:w for i,w in weights.items() if w})
    seen=set();reader=pq.ParquetFile(path);round_reader=pq.ParquetFile(round_path)
    if reader.num_row_groups!=round_reader.num_row_groups:raise RuntimeError('Restoration and round batch counts differ')
    for group in range(reader.num_row_groups):
        table=reader.read_row_group(group);ci=column_array(table,'physical_index');firms=column_array(table,'firm_id')
        unique_firms=np.unique(firms);B=len(unique_firms)
        if set(np.unique(ci).tolist())!=set(range(C)) or len(table)!=C*B:
            raise RuntimeError('Restoration row group is not a complete condition×firm batch')
        fi=np.searchsorted(unique_firms,firms)
        if len(np.unique(ci*B+fi))!=C*B or seen.intersection(map(int,unique_firms)):
            raise RuntimeError('Duplicate firm or condition record in restoration data')
        seen.update(map(int,unique_firms))
        if np.any(column_array(table,'base_index')!=cell['base_index']):raise RuntimeError('Restoration base-index mismatch')
        cat=column_array(table,'category_by_definition')
        # S3B-A03: maintenance requires the entire post-shock window, while
        # category1 records only no immediate material loss at round21.
        rounds=unpack_compact_table(round_reader.read_row_group(group))
        rci=column_array(rounds,'physical_index');rfirms=column_array(rounds,'firm_id')
        if len(rounds)!=len(table) or not np.array_equal(np.unique(rfirms),unique_firms):
            raise RuntimeError('Round batch and restoration firm sets differ')
        rfi=np.searchsorted(unique_firms,rfirms)
        if set(np.unique(rci).tolist())!=set(range(C)) or len(np.unique(rci*B+rfi))!=C*B:
            raise RuntimeError('Malformed round batch rectangle')
        q_tensor=np.empty((C,B,40));q_tensor[rci,rfi]=column_array(rounds,'Q_1_40')
        q=q_tensor[ci,fi];base=column_array(table,'baseline_accuracy')
        affected=column_array(table,'actual_shocked_domain_count')>0
        if not np.array_equal(q[:,20],column_array(table,'Q21')):raise RuntimeError('Round21 score differs between retained tables')
        for d,definition in enumerate(RESTORATION_DEFINITIONS):
            baseline=base[:,0 if d<3 else 1];delta=definition['tolerance'];threshold=baseline-delta
            full=affected&(q[:,20:40]>=threshold[:,None]).all(axis=1)
            late=(cat[:,d]==1)&(q[:,21:40]<threshold[:,None]).any(axis=1)
            mismatch=affected&(((baseline-q[:,20])>delta)!=(q[:,20]<threshold))
            np.add.at(maintained[:,d],ci,full);np.add.at(delayed[:,d],ci,late)
            np.add.at(threshold_disagreement[:,d],ci,mismatch)
        starts=column_array(table,'start_round_by_definition');confirms=column_array(table,'confirmation_round_by_definition')
        if np.any((cat<0)|(cat>3)):raise RuntimeError('Unknown restoration category')
        restored=cat==2
        if np.any(confirms[restored]-starts[restored]!=2) or np.any(confirms[restored]>40) or np.any(starts[restored]<21):
            raise RuntimeError('Invalid restoration timing')
        if np.any(starts[~restored]!=0) or np.any(confirms[~restored]!=0):raise RuntimeError('Nonrestored timing sentinel lost')
        for k in range(4):np.add.at(counts[:,:,k],ci,cat==k)
        np.add.at(baselines,ci,column_array(table,'baseline_accuracy'))
        np.add.at(initial_loss,ci,column_array(table,'initial_loss_by_baseline'))
        np.add.at(q21,ci,column_array(table,'Q21'));np.add.at(q40,ci,column_array(table,'Q40'))
        np.add.at(start_sums,ci,starts);np.add.at(confirm_sums,ci,confirms)
        improved=column_array(table,'Q40_improved_from_Q21')[:,None]
        np.add.at(improved_loss_unrestored,ci,improved&(cat==3))
        np.add.at(improved_any_unrestored,ci,improved&(cat!=2))
        tensor=np.empty((C,B,6),dtype=np.uint8);tensor[ci,fi]=cat
        indicators=tensor[:,:,:,None]==np.arange(4)[None,None,None,:]
        for k,weights in enumerate(coefficients):
            vector=np.zeros((B,6,4),dtype=np.float64)
            for i,weight in weights.items():vector+=weight*indicators[i]
            sums[k]+=vector.sum(axis=0);squares[k]+=(vector*vector).sum(axis=0)
    if seen!=set(range(1,N+1)) or np.any(counts.sum(axis=2)!=N):
        raise RuntimeError('Restoration histories do not match full frozen N')
    rows=[];paired_rows=[]
    for i,condition in enumerate(cell['unique_conditions']):
        for d,definition in enumerate(RESTORATION_DEFINITIONS):
            n0,n1,n2,n3=map(int,counts[i,d]);b=0 if d<3 else 1
            ms=float(start_sums[i,d]/n2) if n2 else None
            mc=float(confirm_sums[i,d]/n2) if n2 else None
            rows.append({**common(cell),'physical_id':str(condition['physical_id']),'physical_index':i,
                'definition_index':d,'baseline_definition':definition['baseline'],'tolerance':definition['tolerance'],
                'N':N,'no_actual_shock_count':n0,'no_initial_material_loss_count':n1,
                'restoration_confirmed_count':n2,'initial_loss_unrestored_count':n3,'initial_material_loss_count':n2+n3,
                'baseline_mean':baselines[i,b]/N,'Q21_mean':q21[i]/N,'Q40_mean':q40[i]/N,
                'initial_loss_mean_full_sample':initial_loss[i,b]/N,'mean_start_round_among_restored':ms,
                'mean_confirmation_round_among_restored':mc,'mean_start_delay_from21_among_restored':ms-21 if n2 else None,
                'mean_confirmation_delay_from21_among_restored':mc-21 if n2 else None,
                'Q40_improved_among_initial_loss_unrestored_count':int(improved_loss_unrestored[i,d]),
                'Q40_improved_among_any_unrestored_count':int(improved_any_unrestored[i,d]),
                'any_unrestored_count':N-n2,'censored_after_initial_loss_count':n3,'censoring_round':40,
                'actual_shock_count':N-n0,'sustained_maintenance_count':int(maintained[i,d]),
                'category1_later_below_benchmark_count':int(delayed[i,d]),
                'sustained_maintenance_proportion_all':float(maintained[i,d]/N),
                'sustained_maintenance_proportion_affected':float(maintained[i,d]/(N-n0)) if N>n0 else None,
                'category1_later_below_proportion_all':float(delayed[i,d]/N),
                'category1_later_below_proportion_affected':float(delayed[i,d]/(N-n0)) if N>n0 else None,
                'initial_loss_threshold_roundoff_disagreement_count':int(threshold_disagreement[i,d]),
                'data_origin':'SIMULATED'})
    for k,contrast in enumerate(paired):
        meta=contrast.get('metadata',{})
        means=sums[k]/N
        variance=(squares[k]-sums[k]*sums[k]/N)/(N-1)
        if np.any(variance < -1e-12):raise RuntimeError('Impossible negative paired category variance')
        variance=np.maximum(variance,0.)
        for d,definition in enumerate(RESTORATION_DEFINITIONS):
            for category in range(4):
                v=float(variance[d,category])
                paired_rows.append({**common(cell),'contrast_id':contrast['contrast_id'],'contrast_family':contrast['family'],
                    'context':meta.get('context'),'learning_rule':meta.get('learning_rule'),'L':meta.get('L'),
                    'definition_index':d,'baseline_definition':definition['baseline'],'tolerance':definition['tolerance'],
                    'category':category,'category_label':RESTORATION_CATEGORIES[category],'N':N,
                    'mean_probability_contrast':float(means[d,category]),'variance':v,'sd':float(np.sqrt(v)),
                    'mcse':float(np.sqrt(v/N)),'terms_json':jdump(contrast['terms']),'data_origin':'SIMULATED'})
    return pa.Table.from_pylist(rows,schema=RESTORATION_SCHEMA),pa.Table.from_pylist(paired_rows,schema=PAIRED_RESTORATION_SCHEMA)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,default=ROOT/'data/production_v2')
    ap.add_argument('--manifest',type=Path,default=ROOT/'config/experiment_manifest.json')
    ap.add_argument('--output',type=Path,default=ROOT/'results/aggregated_v1')
    args=ap.parse_args();args.run=args.run.resolve();args.manifest=args.manifest.resolve();args.output=args.output.resolve()
    manifest=load(args.manifest);run=verify_completed_run(args.run,args.manifest,manifest);N=int(run['N'])
    if args.output.exists():raise RuntimeError('Preserve existing aggregation; choose a documented new version')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_name(args.output.name+'.inprogress_'+uuid.uuid4().hex[:8]);temp.mkdir()
    start=time.perf_counter();writers={};counts={};input_records=[]
    report={'status':'RUNNING','started_at_utc':now(),'source_run_manifest_sha256':sha(args.run/'run_manifest.json'),
            'experiment_manifest_sha256':sha(args.manifest),'run_signature':run['run_signature'],'N':N,
            'script_sha256':sha(Path(__file__)),'outcome_module_sha256':sha(ROOT/'src/experiment_outputs.py'),
            'storage_reader_sha256':sha(ROOT/'src/storage_v2.py'),
            'maintenance_amendment_sha256':sha(ROOT/'config/maintenance_reporting_amendment.json'),
            'data_origin':'SIMULATED','new_simulations':0,'restoration_contrast_families':sorted(RESTORATION_CONTRAST_FAMILIES),
            'restoration_variance':'Within-firm full category-indicator contrast, then sample variance acrossfirms',
            'completed_base_indices':[]}
    write_json(temp/'aggregation_manifest.json',report)
    try:
        base_rows=[]
        for cell in manifest['base_cells']:
            base_rows.append({**common(cell),**cell['parameters'],'canonical_parameters':cell['canonical_parameters'],
                              'data_origin':'SIMULATED'})
        base_table=pa.Table.from_pylist(base_rows)
        pq.write_table(base_table,temp/'base_cells.parquet',compression='zstd',compression_level=3)
        for cell in manifest['base_cells']:
            cell_dir=args.run/'cells'/f'cell_{cell["base_index"]:04d}'
            record=verify_input_cell(cell_dir,cell,run)
            contrast,condition=aggregate_primary(cell_dir,cell,N)
            restoration,paired=aggregate_restoration(cell_dir/'restoration.parquet',cell,N,cell_dir/'compact_rounds.parquet')
            for name,table in [('contrast_wide',contrast),('condition_wide',condition),
                               ('restoration_summary',restoration),('paired_restoration_contrasts',paired)]:
                if name not in writers:writers[name]=parquet_writer(temp/(name+'.parquet'),table.schema);counts[name]=0
                writers[name].write_table(table,row_group_size=len(table));counts[name]+=len(table)
            input_records.append({'base_index':cell['base_index'],'cell_manifest_sha256':sha(cell_dir/'cell_manifest.json'),
                                  'verified_files':[x for x in record['files'] if x['name'] in
                                      ['condition_summary.parquet','contrast_summary.parquet','restoration.parquet','compact_rounds.parquet']]})
            report['completed_base_indices'].append(cell['base_index'])
            if len(input_records)%20==0:print(json.dumps({'completed':len(input_records),'total':876,'elapsed_seconds':time.perf_counter()-start}),flush=True)
        for writer in writers.values():writer.close()
        writers={}
        expected={'contrast_wide':sum(len(c['contrasts']) for c in manifest['base_cells']),
                  'condition_wide':sum(len(c['conditions']) for c in manifest['base_cells']),
                  'restoration_summary':sum(len(c['unique_conditions'])*6 for c in manifest['base_cells']),
                  'paired_restoration_contrasts':sum(sum(x['family'] in RESTORATION_CONTRAST_FAMILIES for x in c['contrasts'])*24
                                                    for c in manifest['base_cells'])}
        if counts!=expected:raise RuntimeError(f'Aggregate row counts mismatch:{counts} versus{expected}')
        write_json(temp/'input_checkpoint_inventory.json',input_records)
        report.update({'status':'PASS','completed_at_utc':now(),'elapsed_seconds':time.perf_counter()-start,
                       'base_cells':876,'output_rows':{'base_cells':len(base_rows),**counts},
                       'files':[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(temp.glob('*.parquet'))]})
        write_json(temp/'aggregation_manifest.json',report);os.replace(temp,args.output)
        print(json.dumps({'status':'PASS','output':str(args.output),'rows':report['output_rows'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    except Exception:
        for writer in writers.values():writer.close()
        report.update({'status':'FAILED','failed_at_utc':now(),'traceback':traceback.format_exc()})
        write_json(temp/'aggregation_manifest.json',report);raise

if __name__=='__main__':main()
