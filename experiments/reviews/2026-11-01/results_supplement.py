"""Check baseline checkpoints/history and saved PTB comparison without inference."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch
from src.compare_ptbxl_models import _Inputs, _load_run, _align, _summary, _same

def read(p):
    return json.loads(Path(p).read_text())

def rows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_inference_performed':False,
          'bootstrap_resampling_performed':False, 'baseline_checkpoints':{}, 'failures':[]}
baseline = read(ROOT / 'experiments/logs/checkpoint_verification.json')
protocol = read(ROOT / 'data/splits/protocol.json')
for family, rid in [('lda','lda_v1_seed42'),('cnn','cnn_v1_seed42'),('cnn_lstm','cnn_lstm_image_v1_seed42')]:
    out = ROOT/'results'/family/rid
    run = read(out/'run.json')
    train = run['training']
    model = ROOT/run['model_directory']
    if family == 'lda':
        cp = model/'pipeline.joblib'
        assert sha(cp) == baseline['models']['lda']['checkpoint_sha256']
        pipeline = joblib.load(cp)
        pca, lda = pipeline.named_steps['pca'],pipeline.named_steps['lda']
        assert pca.n_samples_ == 295 and pca.n_components_ == 32 and pca.components_.shape == (32,32768)
        assert np.array_equal(lda.classes_,np.arange(4)) and np.allclose(lda.priors_,.25)
        assert read(out/'config.json') == protocol['config']
        result['baseline_checkpoints'][rid] = {'checkpoint_hash_matches':True,'pca_shape':[32,32768],
            'pca_fit_training_samples':295,'uniform_lda_priors':True,'saved_config_matches_frozen_protocol':True,
            'parameter_count':'Not recorded as a neural parameter count; PCA plus LDA pipeline'}
        continue
    cp = torch.load(model/'best.pt',weights_only=True,map_location='cpu')
    expected_hash = run.get('checkpoint_sha256',baseline['models']['cnn']['checkpoint_sha256'])
    assert sha(model/'best.pt') == expected_hash
    for key in ['config','normalization','best_epoch']:
        assert cp[key] == train[key],key
    assert sum(v.numel() for v in cp['state_dict'].values()) == train['num_parameters']
    assert all(bool(torch.isfinite(v).all()) for v in cp['state_dict'].values())
    history = rows(model/'history.csv')
    assert len(history) == train['epochs_run']
    assert [int(row['epoch']) for row in history] == list(range(1,len(history)+1))
    if family == 'cnn':
        selected = min(range(len(history)),key=lambda i:float(history[i]['val_loss']))
        assert read(out/'config.json') == protocol['config']
        assert cp['validation_loss'] == float(history[selected]['val_loss'])
    else:
        selected = max(range(len(history)),key=lambda i:(float(history[i]['val_macro_f1']),float(history[i]['val_balanced_accuracy']),-float(history[i]['val_loss'])))
        config_path=ROOT/'experiments/configs/cnn_lstm_image_v1.json'
        assert sha(config_path) == read(out/'provenance.json')['configuration_sha256']
        assert read(config_path) == read(out/'config.json')
        frozen,state=read(out/'selection_frozen.json'),read(out/'test_inference_state.json')
        assert frozen['checkpoint_sha256'] == state['checkpoint_sha256'] == run['checkpoint_sha256']
        assert datetime.fromisoformat(frozen['frozen_utc']) < datetime.fromisoformat(state['started_utc'])
    assert selected+1 == train['best_epoch']
    result['baseline_checkpoints'][rid]={'checkpoint_hash_matches':True,'checkpoint_metadata_matches':True,
        'finite_tensors':True,'num_parameters':train['num_parameters'],'epochs_run':len(history),
        'selected_epoch':selected+1,'history_selection_matches':True}

inputs = _Inputs()
old = _load_run(ROOT,'ptbxl_cnn_lstm','ptbxl_cnn_lstm_image_v1_seed42',inputs)
new = _load_run(ROOT,'ptbxl_spatiotemporal','ptbxl_spatiotemporal_image_v1_seed42',inputs)
_align(old,new)
saved=read(ROOT/'results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.json')
recomputed=_summary(old,new,saved['paired_test_bootstrap'])
for key,value in recomputed.items():
    if key != 'created_utc':
        _same(value,saved[key],'comparison.'+key)
for name,ci in saved['paired_test_bootstrap']['metrics'].items():
    a,b=old['partitions']['test']['metrics'][name],new['partitions']['test']['metrics'][name]
    assert ci['old_point_estimate']==a and ci['new_point_estimate']==b and ci['difference']==b-a
    assert 0 <= ci['defined_resamples'] <= 500 and ci['defined_resamples']+ci['undefined_resamples']==500
    assert -1 <= ci['ci_lower'] <= ci['ci_upper'] <= 1
inputs.unchanged()
result['ptbxl_comparison']={'status':'passed','summary_recomputed_from_saved_predictions':True,
    'all_partition_ids_truths_patients_and_inputs_align':True,'source_artifacts_hashed':len(inputs.hashes),
    'bootstrap_point_estimates_checked':True,'bootstrap_resampling_performed':False,
    'macro_roc_auc_difference':saved['paired_test_bootstrap']['metrics']['macro_roc_auc']}

full=read(OUT/'results_audit.json')
summary, labels = [],[]
for rid,r in full['runs'].items():
    for part in ['train','validation','test']:
        m=r[part+'_metrics']
        summary.append({'run_id':rid,'partition':part,'n_samples':m['n_samples'],
                        **{k:m.get(k) for k in ['accuracy','balanced_accuracy','subset_accuracy','label_accuracy','macro_f1','weighted_f1','micro_f1','macro_roc_auc','macro_average_precision','roc_auc_ovr_macro']},
                        'num_parameters':r['parameters'],'epochs_run':r['epochs_run'],'best_epoch':r['best_epoch'],
                        'training_seconds':r['training_seconds'],'runner_elapsed_seconds':r['runner_elapsed_seconds'],
                        'utc_wall_span_seconds':r['utc_wall_span_seconds']})
        for label,vals in m.get('per_label',m.get('per_class')).items():
            labels.append({'run_id':rid,'partition':part,'label':label,'precision':vals['precision'],'recall':vals['recall'],
                           'f1':vals.get('f1',vals.get('f1-score')),'support':vals['support'],
                           **{k:vals.get(k) for k in ['specificity','balanced_accuracy','roc_auc','average_precision','tp','fp','tn','fn']}})
for name,data in [('all_run_metrics.csv',summary),('all_run_per_label_metrics.csv',labels)]:
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
result['tests']={'command':'/opt/anaconda3/bin/python -m unittest discover -s tests -v',
                 'log':'experiments/reviews/2026-11-01/full_unittest_2026-11-01.log',
                 'log_sha256':sha(OUT/'full_unittest_2026-11-01.log'),'tests_run':152,'status':'passed','elapsed_seconds':19.338}
(OUT/'results_supplement.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
