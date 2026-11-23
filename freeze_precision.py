"""Select the one global production N from completed paired variance estimates.

No effect direction enters this decision. Execution requires an independently
verified pilot; the output records actual probe sizes before production starts.
"""
from pathlib import Path
import sys,json,hashlib,math,shutil
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_log import append_process,append_decision
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    target=ROOT/'config/precision_freeze.json'
    if target.exists():raise RuntimeError('Refusing to overwrite a precision freeze')
    pilot=ROOT/'data/variance_v1';run=json.loads((pilot/'run_manifest.json').read_text())
    audit=json.loads((ROOT/'audit/variance_precision_review.json').read_text())
    if run['status']!='PASS' or audit['status']!='PASS':raise RuntimeError('Pilot and independent review must pass')
    cells=sorted((pilot/'cells').glob('cell_*'))
    if len(cells)!=876:raise RuntimeError('Incomplete variance design')
    rows=[];probes=[];totalbytes=0;elapsed=0;maximum=None;summary_count=0
    for cell in cells:
        m=json.loads((cell/'cell_manifest.json').read_text());elapsed+=m['elapsed_seconds']
        if m['N']!=128 or m['namespace']!=12:raise RuntimeError('Wrong pilot N or namespace')
        t=pq.read_table(cell/'contrast_summary.parquet',columns=['contrast_id','key','variance','precision_primary'])
        t=t.filter(t['precision_primary']);v=t['variance'].to_numpy();j=int(np.argmax(v));record=t.slice(j,1).to_pylist()[0]
        record.update(base_index=m['base_index'],family=m['family'],cell_key=m['cell_key'],N_pilot=128)
        record['required_N_unrounded']=record['variance']/0.005**2
        rows.append(record);summary_count+=len(t)
        if maximum is None or record['variance']>maximum['variance']:maximum=record
        files={x['name']:x['bytes'] for x in m['files']};totalbytes+=sum(files.values())
        if m['storage_probe_only']:
            probes.append({'base_index':m['base_index'],'family':m['family'],'conditions':m['physical_conditions'],
                'pilot_N':128,'round_bytes':files['compact_rounds.parquet'],
                'phase_bytes':files['phase_vectors.parquet'],
                'round_bytes_per_condition_firm':files['compact_rounds.parquet']/(128*m['physical_conditions']),
                'phase_bytes_per_condition_firm':files['phase_vectors.parquet']/(128*m['physical_conditions'])})
    uncapped=64*math.ceil(max(256,maximum['variance']/0.005**2)/64)
    n=min(8192,uncapped);physical=79164;logical=104400
    # Conservative projection from the largest observed per-history file size.
    # Restoration uses six definitions plus metadata, provisioned at 256 bytes.
    round_est=math.ceil(max(x['round_bytes_per_condition_firm'] for x in probes)*physical*n)
    phase_est=math.ceil(max(x['phase_bytes_per_condition_firm'] for x in probes)*physical*n)
    restoration_est=256*physical*n
    summary_est=totalbytes-sum((p/'phase_vectors.parquet').stat().st_size for p in cells)-sum(p['round_bytes'] for p in probes)
    production_est=round_est+phase_est+restoration_est+summary_est
    free=shutil.disk_usage(ROOT).free
    # An archive copy and 2 GB headroom are included. No silent grid reduction.
    required=2*production_est+2_000_000_000
    feasible=required<free
    record={'status':'FROZEN' if feasible else 'RESOURCE_HOLD','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selected_N':n,'uncapped_N':uncapped,'cap_reached':uncapped>8192,'target_mcse':0.005,
        'min_N':256,'max_N':8192,'block_size':64,'pilot_N':128,'pilot_namespace':12,'production_namespace':21,
        'master_seed':20261121,'selection_formula':'min(8192,64*ceil(max(256,max_paired_variance/0.005**2)/64))',
        'maximum_paired_variance':maximum['variance'],'maximum_case':maximum,
        'expected_mcse_at_selected_N':float(np.sqrt(maximum['variance']/n)),
        'registered_primary_variance_estimates':summary_count,'base_cells':876,
        'logical_conditions':logical,'physical_conditions':physical,
        'paired_histories':876*n,'logical_trajectories':logical*n,'physical_trajectories':physical*n,
        'logical_domain_round_evaluations':logical*n*1200,'physical_domain_round_evaluations':physical*n*1200,
        'manifest_sha256':sha(ROOT/'config/experiment_manifest.json'),
        'pilot_run_manifest_sha256':sha(pilot/'run_manifest.json'),
        'independent_precision_review_sha256':sha(ROOT/'audit/variance_precision_review.json'),
        'pilot_total_file_bytes':totalbytes,'pilot_sum_cell_seconds':elapsed,
        'transition_time_projection_seconds':elapsed*n/128,
        'storage_probes':probes,'resource_projection':{
            'round_bytes':round_est,'phase_bytes':phase_est,'restoration_bytes':restoration_est,
            'summary_bytes':summary_est,'production_total_bytes':production_est,
            'free_bytes_before_production':free,'production_plus_archive_and_headroom_bytes':required,
            'feasible':feasible,'interpretation':'Conservative projection; actual Parquet compression varies by cell. Domain traces regenerated, not retained.'},
        'memory_strategy':'128-firm batches; one cell in memory; full design manifest plus bounded scores and paired phase vectors. No full domain dataset.',
        'production_authorized':feasible,'production_results_seen':False,
        'parameter_or_structural_uncertainty_covered':False,'data_origin':'SIMULATED'}
    (ROOT/'results/variance_cell_maxima.json').write_text(json.dumps(rows,indent=2))
    target.write_text(json.dumps(record,indent=2))
    append_process('Complete independent variance pilot and freeze one global production N',
        f'876 cells; {summary_count} registered paired variance estimates; maximum={maximum["variance"]:.12g}; N={n}; feasibility={feasible}.',
        'Production may start only when the recorded resource projection passes.',namespace=12,
        inputs={'pilot_run_manifest':record['pilot_run_manifest_sha256'],'independent_review':record['independent_precision_review_sha256']},
        affected=['config/precision_freeze.json','results/variance_cell_maxima.json'])
    append_decision(f'Freeze global N={n} for every production cell',
        'Exact E17 maximum paired-variance rule; no use of effect direction. The same N applies to all registered contrasts.',
        category='Prospective precision decision',affected=['config/precision_freeze.json'])
    print(json.dumps({k:record[k] for k in ['status','selected_N','uncapped_N','maximum_case','resource_projection','production_authorized']},indent=2))
if __name__=='__main__':main()
