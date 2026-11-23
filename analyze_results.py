"""Transparent design-point classifications and numerical-uncertainty summaries.

All summaries concern the registered hypothetical parameter design, not a
sample of family firms. No empirical significance tests are performed.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,argparse
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_outputs import OUTCOME_KEYS
YKEYS=['early__Q','post_shock__Q','post_shock__C_post','withdrawal__C_post','withdrawal__H_post','post_shock__R']
PARAMS=['kappa_0','c','q_H','r_AI','epsilon_AI','v_check','c_AI','c_V','a','chi','alpha','beta','eta','phi','q_V','gamma','q_0','lambda_q','d']

def material(x,delta):
    tol=1e-12 if delta==0 else 0.
    return np.where(x>delta+tol,1,np.where(x< -delta-tol,-1,0))
def numerical(x,se,delta):
    tol=1e-12 if delta==0 else 0.
    lo=x-1.96*se;hi=x+1.96*se
    return np.where(lo>delta+tol,1,np.where(hi< -delta-tol,-1,
        np.where((lo>=-delta-tol)&(hi<=delta+tol),0,2)))
def summaries(df,groups,keys=YKEYS):
    rows=[]
    for values,g in df.groupby(groups,dropna=False,sort=True):
        if not isinstance(values,tuple):values=(values,)
        base=dict(zip(groups,values))
        for key in keys:
            x=g[key].to_numpy();se=g['se__'+key].to_numpy()
            for delta in [.02,0.,.05]:
                s=material(x,delta);n=numerical(x,se,delta)
                rows.append({**base,'outcome':key,'threshold':delta,'design_points':len(g),
                    'mean_across_design':float(np.mean(x)),'median':float(np.median(x)),
                    'minimum':float(np.min(x)),'maximum':float(np.max(x)),
                    'positive':int(np.sum(s==1)),'negative':int(np.sum(s==-1)),'within_threshold':int(np.sum(s==0)),
                    'confirmed_positive':int(np.sum(n==1)),'confirmed_negative':int(np.sum(n==-1)),
                    'confirmed_negligible':int(np.sum(n==0)),'boundary_inconclusive':int(np.sum(n==2)),
                    'max_mcse':float(np.max(se)),'data_origin':'SIMULATED'})
    return pd.DataFrame(rows)
def normalise(df):
    if 'base_family' in df and 'family' not in df:df=df.rename(columns={'base_family':'family'})
    if 'block' in df and 'context' not in df:df=df.rename(columns={'block':'context'})
    return df

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--primary-source',type=Path,default=ROOT/'results/aggregate');args=parser.parse_args()
    started=time.perf_counter();out=ROOT/'results/analysis';out.mkdir(parents=True,exist_ok=True)
    source=args.primary_source.resolve()
    base=normalise(pd.read_parquet(source/'base_cells.parquet'))
    contrasts=normalise(pd.read_parquet(source/'contrast_wide.parquet'))
    if len(contrasts)!=200472 or len(base)!=876:raise RuntimeError('Incomplete aggregate')
    missing=[x for x in PARAMS if x not in contrasts]
    contrasts=contrasts.merge(base[['base_index']+missing],on='base_index',validate='many_to_one')
    contrasts['access_direction']=np.where(contrasts.r_AI>.6,'increased',np.where(contrasts.r_AI<.6,'decreased','matched'))
    contrasts['premium_sign']=np.where(contrasts.a>0,'positive',np.where(contrasts.a<0,'negative','zero'))
    contrasts['base_displacement_sign']=np.where(contrasts.chi>0,'positive',np.where(contrasts.chi<0,'negative','zero'))
    contrasts['comparison']=contrasts['left'].fillna('')+'-'+contrasts['right'].fillna('')
    contrasts.loc[contrasts.contrast_family=='principal_interaction','comparison']='Psi_access_source'
    # Full precision coverage, including all20 outcome columns and all contrasts.
    precision=[]
    for key in OUTCOME_KEYS:
        se=contrasts['se__'+key];idx=se.idxmax()
        precision.append({'outcome':key,'estimates':len(se),'mcse_max':float(se.max()),
            'mcse_median':float(se.median()),'above_target':int((se>.005+1e-15).sum()),
            'max_base_index':int(contrasts.loc[idx,'base_index']),
            'max_contrast':contrasts.loc[idx,'contrast_id'],'data_origin':'SIMULATED'})
    pd.DataFrame(precision).to_parquet(out/'numerical_precision.parquet',index=False)
    # P1: exhaustive nine regions for each of the two later horizons.
    select=((contrasts.contrast_family=='core_mode')&contrasts.comparison.isin(['C-B','D-Bplus']))|(contrasts.contrast_family=='pure_access')
    p1=contrasts[select].copy();p1rows=[]
    for delta in [.02,0.,.05]:
        for horizon in ['post_shock__C_post','withdrawal__C_post']:
            z=p1[['base_index','family','context','comparison','learning_rule','L','access_direction','a','chi','r_AI','c_AI','epsilon_AI']].copy()
            z['threshold']=delta;z['horizon']=horizon
            z['early_accuracy_difference']=p1['early__Q'];z['early_mcse']=p1['se__early__Q']
            z['competence_difference']=p1[horizon];z['competence_mcse']=p1['se__'+horizon]
            x=material(z.early_accuracy_difference,delta);y=material(z.competence_difference,delta)
            xn=numerical(z.early_accuracy_difference,z.early_mcse,delta);yn=numerical(z.competence_difference,z.competence_mcse,delta)
            names={1:'better',0:'within threshold',-1:'worse'}
            z['region']=[f'Assisted {names[a]}; independent {names[b]}' for a,b in zip(x,y)]
            z['dissociation']=(x==1)&(y==-1)
            z['confirmed_dissociation']=(xn==1)&(yn==-1)
            z['boundary_inconclusive']=(xn==2)|(yn==2)
            z['data_origin']='SIMULATED';p1rows.append(z)
    p1long=pd.concat(p1rows,ignore_index=True)
    p1long.to_parquet(out/'P1_regions.parquet',index=False)
    groups=['family','context','comparison','learning_rule','L','access_direction','threshold','horizon']
    p1sum=p1long.groupby(groups,dropna=False).agg(design_points=('base_index','size'),dissociation=('dissociation','sum'),
        confirmed_dissociation=('confirmed_dissociation','sum'),boundary_inconclusive=('boundary_inconclusive','sum')).reset_index()
    p1sum['dissociation_share']=p1sum.dissociation/p1sum.design_points
    p1sum['confirmed_share']=p1sum.confirmed_dissociation/p1sum.design_points
    p1sum['data_origin']='SIMULATED';p1sum.to_parquet(out/'P1_region_summary.parquet',index=False)
    p1long.groupby(groups+['region'],dropna=False).size().rename('count').reset_index().to_parquet(out/'P1_all_region_counts.parquet',index=False)
    union_groups=['base_index','family','context','comparison','learning_rule','L','access_direction','threshold']
    union=p1long.groupby(union_groups,dropna=False).agg(dissociation_either=('dissociation','max'),
        confirmed_either=('confirmed_dissociation','max'),any_boundary_inconclusive=('boundary_inconclusive','max')).reset_index()
    union['data_origin']='SIMULATED';union.to_parquet(out/'P1_either_horizon_points.parquet',index=False)
    union_summary=union.groupby(union_groups[1:],dropna=False).agg(design_points=('base_index','size'),
        dissociation_either=('dissociation_either','sum'),confirmed_either=('confirmed_either','sum'),
        any_boundary_inconclusive=('any_boundary_inconclusive','sum')).reset_index()
    union_summary['data_origin']='SIMULATED';union_summary.to_parquet(out/'P1_either_horizon_summary.parquet',index=False)
    # Principal interaction, structural removals, source attribution and checking.
    psi=contrasts[contrasts.contrast_family=='principal_interaction'].copy()
    psi.to_parquet(out/'principal_Psi_points.parquet',index=False)
    summaries(psi,['family','context','learning_rule','access_direction'],YKEYS).to_parquet(out/'principal_Psi_summary.parquet',index=False)
    mechanism=contrasts[contrasts.contrast_family=='mechanism_interaction']
    summaries(mechanism,['family','context','learning_rule','access_direction'],YKEYS).to_parquet(out/'mechanism_Psi_summary.parquet',index=False)
    sourceeffects=contrasts[contrasts.contrast_family=='source']
    summaries(sourceeffects,['family','context','interface','learning_rule','premium_sign','base_displacement_sign'],YKEYS).to_parquet(out/'P2_source_summary.parquet',index=False)
    checking=contrasts[(contrasts.contrast_family=='core_mode')&contrasts.comparison.isin(['D-C','Bplus-B','Aplus-A1'])]
    checking.to_parquet(out/'P3_checking_points.parquet',index=False)
    summaries(checking,['family','context','comparison','learning_rule','L'],YKEYS).to_parquet(out/'P3_checking_summary.parquet',index=False)
    component=contrasts[contrasts.contrast_family.isin(['component','component_interaction'])]
    summaries(component,['family','component_label','learning_rule','L'],YKEYS).to_parquet(out/'factorial_summary.parquet',index=False)
    # Structural controls use complete saved paired vectors; exact zero expected.
    controls=[]
    def control(name,df,keys):
        maximum=max(float(df[k].abs().max()) if len(df) else 0 for k in keys)
        se_max=max(float(df['se__'+k].abs().max()) if len(df) else 0 for k in keys)
        controls.append({'control':name,'rows':len(df),'outcomes':len(keys),'maximum_absolute_contrast':maximum,
            'maximum_mcse':se_max,'status':'PASS' if maximum<1e-12 and se_max<1e-12 else 'FAIL'})
    control('Matched AI/archive identities in focused grid',contrasts[(contrasts.family=='focused')&contrasts.comparison.isin(['C-B','D-Bplus'])],OUTCOME_KEYS)
    control('Zero source premium removes label effects',sourceeffects[sourceeffects.a==0],OUTCOME_KEYS)
    independent=[k for k in OUTCOME_KEYS if '__C_' in k or '__H_post' in k or 'competence_deficit' in k]
    control('chi0 constant learning: label invariant independent competence',sourceeffects[(sourceeffects.learning_rule=='constant_eta')&((sourceeffects.chi==0)|sourceeffects.context.isin(['chi_zero','chi_zero_reliability_frozen','chi_zero_gamma_zero']))],independent)
    control('chi0 constant learning: pure-access/source interaction zero for competence',psi[(psi.learning_rule=='constant_eta')&psi.context.isin(['chi_zero','chi_zero_reliability_frozen','chi_zero_gamma_zero'])],independent)
    pd.DataFrame(controls).to_parquet(out/'structural_controls.parquet',index=False)
    # Descriptive response surfaces, explicitly exploratory summaries of a
    # pre-specified model response, never empirical coefficients or p-values.
    meta=[];coef=[]
    interactions=[('q_H','d'),('a','chi'),('kappa_0','r_AI'),('lambda_q','phi'),('q_V','c_V'),('r_AI','c_AI'),('r_AI','epsilon_AI'),('c_AI','epsilon_AI')]
    selected=psi[psi.context=='base']
    for (fam,rule),g in selected.groupby(['family','learning_rule']):
        raw=g[PARAMS].to_numpy();means=raw.mean(axis=0);sd=raw.std(axis=0);z=(raw-means)/sd
        X=np.column_stack([np.ones(len(g)),z]+[z[:,PARAMS.index(a)]*z[:,PARAMS.index(b)] for a,b in interactions])
        names=['intercept']+PARAMS+[a+' x '+b for a,b in interactions]
        for key in ['early__Q','post_shock__C_post','withdrawal__C_post']:
            y=g[key].to_numpy();b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b
            denom=float(np.sum((y-y.mean())**2));r2=1-float(np.sum(res**2))/denom if denom>1e-28 else None
            meta.append({'family':fam,'learning_rule':rule,'outcome':key,'design_points':len(y),'terms':len(names),
                'R_squared_descriptive':r2,'RMSE':float(np.sqrt(np.mean(res**2))),'role':'Exploratory deterministic response-surface summary, no empirical inference'})
            coef.extend({'family':fam,'learning_rule':rule,'outcome':key,'term':name,'coefficient':float(value)} for name,value in zip(names,b))
    pd.DataFrame(meta).to_parquet(out/'metamodel_fit.parquet',index=False)
    pd.DataFrame(coef).to_parquet(out/'metamodel_coefficients.parquet',index=False)
    # Static oracle comparisons become available from the separate registered
    # analytical script. Means and complete paired MCSEs retain distinct roles.
    oracle_path=out/'oracle_principal.parquet'
    oracle_record=None
    if oracle_path.exists():
        oracle=pd.read_parquet(oracle_path)
        joined=psi.merge(oracle[['base_index','contrast_id']+list(OUTCOME_KEYS)],on=['base_index','contrast_id'],suffixes=('','__oracle'),validate='one_to_one')
        for k in OUTCOME_KEYS:
            joined['residual__'+k]=joined[k]-joined[k+'__oracle']
            joined['z__'+k]=np.divide(joined['residual__'+k],joined['se__'+k],out=np.zeros(len(joined)),where=joined['se__'+k]>1e-12)
        joined.to_parquet(out/'oracle_Psi_comparison.parquet',index=False)
        oracle_record={'rows':len(joined),'max_abs_error':max(float(joined['residual__'+k].abs().max()) for k in OUTCOME_KEYS),
            'max_abs_z':max(float(joined['z__'+k].abs().max()) for k in OUTCOME_KEYS),
            'max_zero_se_residual':max(float(joined.loc[joined['se__'+k]<=1e-12,'residual__'+k].abs().max()) if (joined['se__'+k]<=1e-12).any() else 0 for k in OUTCOME_KEYS),
            'interpretation':'Monte Carlo residuals around exact expectations; not exact equality of random trajectories.'}
    report={'status':'PASS' if all(x['status']=='PASS' for x in controls) else 'CONTROL_FAILURE',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'data_origin':'SIMULATED',
        'primary_source':str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        'base_cells':len(base),'registered_contrasts':len(contrasts),'primary_estimates':len(contrasts)*20,
        'numerical_precision':precision,'structural_controls':controls,'oracle_comparison':oracle_record,
        'classification':'Complete parameter-design point counts; pointwise Monte Carlo intervals, not population prevalence or simultaneous inference.',
        'exploratory_analyses':['Descriptive standardized OLS response surfaces, explicitly marked exploratory'],
        'elapsed_seconds':time.perf_counter()-started}
    (ROOT/'audit/analysis_computation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['numerical_precision','structural_controls']},indent=2))
if __name__=='__main__':main()
