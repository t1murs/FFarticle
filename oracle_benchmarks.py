"""Evaluate registered fixed-reliance analytical benchmarks without random draws."""
from pathlib import Path
import sys,json,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference_verification/src'))
sys.path.insert(0,str(ROOT/'src'))
from family_ai.oracle import coefficients, population_paths
from experiment_outputs import phase_vectors,OUTCOME_KEYS

def exact_scores(p,mode,L,rule):
    types=np.asarray([(truth,error,cov,shock) for truth in (0,1) for error in (0,1)
                      for cov in (0,1) for shock in (0,1)],dtype=np.int8)
    truth,error,cov,shock=types.T;h=truth^error
    weights=.5*np.where(error,1-p['q_H'],p['q_H'])*np.where(cov,p['c'],1-p['c'])*np.where(shock,p['d'],1-p['d'])
    b=p['kappa_0']*truth+(1-p['kappa_0'])/2
    phase_scores=[]
    for phase in range(3):
        theta=truth^(shock*(phase>0))
        c=coefficients(p,mode,L,rule,theta,h,cov,available=phase<2)
        k=c['K'];target=np.divide(c['target_numerator'],k,out=np.zeros_like(b),where=k!=0)
        rate=1-p['alpha']*k
        before=target+(b-target)*rate[None,:]**np.arange(20)[:,None]
        after=target+(b-target)*rate[None,:]**np.arange(1,21)[:,None]
        before=np.where(k==0,b,before);after=np.where(k==0,b,after)
        decision=c['p_slope']*before+c['p_intercept']
        score=np.stack([theta*decision+(1-theta)*(1-decision),
            theta*before+(1-theta)*(1-before),theta*after+(1-theta)*(1-after),
            truth*after+(1-truth)*(1-after),np.broadcast_to(c['resource'],before.shape)],axis=-1)
        phase_scores.append((score*weights[None,:,None]).sum(axis=1));b=after[-1]
    return np.concatenate(phase_scores)

def main():
    started=time.perf_counter()
    run=json.loads((ROOT/'data/production_v2/run_manifest.json').read_text())
    if run['status']!='PASS':raise RuntimeError('Production incomplete')
    m=json.loads((ROOT/'config/experiment_manifest.json').read_text())
    rows=[];psi=[];checks=[]
    contexts={'gamma_zero','chi_zero_gamma_zero','reliability_frozen','chi_zero_reliability_frozen'}
    for cell in m['base_cells']:
        expected={};cache={}
        for c in cell['conditions']:
            if cell['family']=='focused':
                if cell['parameters']['gamma']!=0:continue
            elif c['block'] not in contexts:continue
            p={**cell['parameters'],**c['parameters']}
            kind='gamma_zero'
            if p['gamma']!=0:
                if p['lambda_q']!=0:raise AssertionError('Oracle requires fixed reliability or fixed reliance')
                p['w_0']=p['w_0']+p['gamma']*(p['q_0']-.5);p['gamma']=0
                kind='lambda_zero_equivalent_intercept'
            key=(c['physical_id'],kind)
            if key not in cache:
                scores=exact_scores(p,c['mode'],c['L'],c['learning_rule'])
                cache[key]=phase_vectors(scores[None,:,None,:])[0,0]
                if len(checks)<16:
                    original=population_paths(p,c['mode'],c['L'],c['learning_rule'])
                    err=max(float(np.max(np.abs(scores[:,j]-original[f]))) for j,f in enumerate(['Q','C_pre','C_post','H_post','R']) if f!='H_post')
                    checks.append({'base_index':cell['base_index'],'condition_id':c['condition_id'],'max_error':err})
            v=cache[key];expected[c['condition_id']]=v
            rows.append({'base_index':cell['base_index'],'family':cell['family'],'condition_id':c['condition_id'],
                'context':c['block'],'interface':c['interface'],'L':c['L'],'learning_rule':c['learning_rule'],
                'benchmark':kind,**{k:float(v[i]) for i,k in enumerate(OUTCOME_KEYS)},'data_origin':'ANALYTICAL_MODEL_EXPECTATION'})
        for contrast in cell['contrasts']:
            if contrast['family']!='principal_interaction':continue
            if not all(t['condition_id'] in expected for t in contrast['terms']):continue
            v=sum(t['weight']*expected[t['condition_id']] for t in contrast['terms'])
            psi.append({'base_index':cell['base_index'],'family':cell['family'],'contrast_id':contrast['contrast_id'],
                **contrast['metadata'],**{k:float(v[i]) for i,k in enumerate(OUTCOME_KEYS)},'data_origin':'ANALYTICAL_MODEL_EXPECTATION'})
    out=ROOT/'results/analysis';out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_parquet(out/'oracle_conditions.parquet',index=False)
    pd.DataFrame(psi).to_parquet(out/'oracle_principal.parquet',index=False)
    review={'status':'PASS' if max(x['max_error'] for x in checks)<1e-12 else 'FAIL',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'analytical_conditions':len(rows),
        'analytical_Psi':len(psi),'new_random_draws':0,'closed_form_vs_unchanged_oracle_checks':checks,
        'elapsed_seconds':time.perf_counter()-started,'role':'Registered analytical benchmark; no new simulation model or empirical evidence'}
    (ROOT/'audit/oracle_benchmark_computation.json').write_text(json.dumps(review,indent=2))
    print(json.dumps({k:v for k,v in review.items() if k!='closed_form_vs_unchanged_oracle_checks'},indent=2))
if __name__=='__main__':main()
