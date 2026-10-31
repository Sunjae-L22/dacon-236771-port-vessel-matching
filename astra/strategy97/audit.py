"""Train/catalog-only structural audit; no test data and no holdout fitting."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'shared/experiments/pinned/build70'))
import cv,opus_lib as L

def main():
    tr,cat=cv.load();fit=tr[cv.folds(tr)!=2]
    llm=L.llm_korean_rows(tr,ROOT/'shared/experiments/pinned/build70/llm_korean_scored.csv')
    parts={'augment_korean':L.augment_korean(tr,0),'augment_corrupt':L.augment_corrupt(tr,0),'augment_rename':L.augment_rename(tr,0),'llm':llm}
    recipe={k:{'rows':len(d),'ko':int(d.vessel_name.map(L.is_hangul).sum()),'vessels':int(d.target_vessel_id.nunique())} for k,d in parts.items()}
    err=pd.read_csv(ROOT/'shared/experiments/llm_rerank/weakness_actionability_errors.csv')
    cn=cat.copy();cn['normalized_name']=cn.vessel_name.fillna('').str.upper().str.replace('[^A-Z0-9]','',regex=True)
    for col in ['flag_name','vessel_type']:cn[col]=cn[col].fillna('')
    known=cn[cn.gross_tonnage.notna()];keys=['gross_tonnage','flag_name','vessel_type'];meta=known.groupby(keys).size();full=known.groupby(keys+['normalized_name']).size()
    fitcatalog=known[known.vessel_id.isin(set(fit.target_vessel_id))];sizes=fitcatalog.set_index(keys).index.map(meta)
    observed=fit.groupby('target_vessel_id').agg(flags=('flag_name','nunique'),types=('vessel_type','nunique'))
    report={'79_per_seed_full_train_recipe':recipe,'79_total_queries':sum(v['rows'] for v in recipe.values()),'79_ko_queries':sum(v['ko'] for v in recipe.values()),'79_known_configuration':{'dictionary':1,'ranker_features':60,'korean_route_models':6,'english_route_models':6,'mixed_language_ranker':True,'augmentation_seeds':[0,1000,2000,5000,6000,7000]},'88_family_known_configuration':{'dictionary':0,'ranker_features':70,'korean_route_models':1,'mixed_language_ranker':False,'llm_ranker_sample_cap':2500,'augmentation_seed':84},'error_rows':len(err),'error_unique_targets':int(err.target_vessel_id.nunique()),'errors_by_category_gt':{f'{a}|{b}':int(v) for (a,b),v in err.groupby(['error_category','gt_evidence']).size().items()},'inside20_gt_ties':int(((err.error_category=='inside_top20')&(err.gt_evidence=='tie')).sum()),'metadata_only_catalog_classes':{'nonmissing_gt_rows':len(known),'ambiguous_class_rows':int(meta[meta>1].sum()),'ambiguous_class_count':int((meta>1).sum()),'largest_class':int(meta.max())},'same_name_and_metadata_classes':{'ambiguous_class_rows':int(full[full>1].sum()),'ambiguous_class_count':int((full>1).sum()),'largest_class':int(full.max())},'outerfit2_catalog_metadata_only_collision':{'fit_catalog_vessels':len(sizes),'ambiguous_fraction':float(np.mean(sizes>1)),'median_class_size':float(np.median(sizes))},'outerfit2_observed_label_variation':{'vessels':len(observed),'multi_flag':int((observed['flags']>1).sum()),'multi_type':int((observed['types']>1).sum())}}
    local=fitcatalog.groupby(keys).size();remaining=fitcatalog.set_index(keys).index.map(meta)-fitcatalog.set_index(keys).index.map(local)+1
    report['outerfit2_metadata_collision_excluding_other_fit_targets']={'fraction':float(np.mean(remaining>1)),'median_class_size':float(np.median(remaining))}
    lookup=cat.set_index('vessel_id');equal={}
    for field in ['gross_tonnage','flag_name','vessel_type','deadweight_tonnage','size','unit']:
        a=lookup.loc[err.target_vessel_id,field].reset_index(drop=True);b=lookup.loc[err.candidate_vessel_id,field].reset_index(drop=True);equal[field]=a.eq(b)|(a.isna()&b.isna())
    same=equal['gross_tonnage']&equal['flag_name']&equal['vessel_type']
    report['error_target_winner_catalog_metadata_equal']={'gt_flag_type':int(same.sum()),'gt_flag_type_dwt_size_unit':int((same&equal['deadweight_tonnage']&equal['size']&equal['unit']).sum()),'among_inside_top20_gt_ties':int((same&err.error_category.eq('inside_top20')&err.gt_evidence.eq('tie')).sum())}
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str));print(json.dumps(report,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':main()
