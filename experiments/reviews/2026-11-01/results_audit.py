"""Read-only audit of five historical runs; write only this review directory."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.metrics import evaluate
from src.baseline_preservation import verify_baseline_preservation


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f))


def compare(a, b, ctx):
    if isinstance(a, dict):
        assert set(a) == set(b), ctx
        for k in a:
            compare(a[k], b[k], ctx + '.' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), ctx
        for i, (aa, bb) in enumerate(zip(a, b)):
            compare(aa, bb, ctx + '.' + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12, (ctx, a, b)
    else:
        assert a == b, (ctx, a, b)


report = {'created_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Saved artifacts only; no historical model inference, training, or bootstrap resampling',
          'audit_script_sha256': sha(__file__), 'failures': [], 'runs': {}, 'hash_manifests': {}}


def checked(name, func):
    print('START', name, flush=True)
    try:
        value = func()
        print('PASS', name, flush=True)
        return value
    except Exception as error:
        details = {'check': name, 'error': repr(error), 'traceback': traceback.format_exc()}
        report['failures'].append(details)
        print('FAIL', name, repr(error), flush=True)
        return details


report['baseline_preservation'] = checked('baseline_preservation', verify_baseline_preservation)
log = rows(ROOT / 'experiments/experiment_log.csv')
report['experiment_log'] = {'sha256': sha(ROOT / 'experiments/experiment_log.csv'),
                            'rows': len(log), 'run_ids': [r['experiment_id'] for r in log]}

for row in log:
    p = ROOT / row['output_directory']
    run = read(p / 'run.json')
    t = run['training']
    model_dir = ROOT / run['model_directory']
    item = {'status': run['status'], 'output_directory': str(p.relative_to(ROOT)),
            'run_sha256': sha(p / 'run.json'), 'model_directory': run['model_directory'],
            'test_metrics': run['metrics']['test'], 'validation_metrics': run['metrics']['validation'],
            'train_metrics': run['metrics']['train'],
            'parameters': t.get('num_parameters'), 'epochs_run': t.get('epochs_run'),
            'best_epoch': t.get('best_epoch'), 'selection_rule': t.get('checkpoint_selection'),
            'training_seconds': t.get('training_seconds', t.get('fit_seconds')),
            'training_module_seconds': t.get('runtime_seconds'),
            'runner_elapsed_seconds': run['total_runtime_seconds'],
            'utc_wall_span_seconds': (datetime.fromisoformat(run['completed_utc']) - datetime.fromisoformat(run['started_utc'])).total_seconds(),
            'test_inference_calls_recorded': run.get('test_inference_calls'),
            'test_policy': run.get('test_evaluation_policy', read(p / 'provenance.json').get('test_policy')),
            'checkpoint_path': str(next(iter(model_dir.glob('best.pt')), model_dir / 'pipeline.joblib').relative_to(ROOT)),
            'failure_or_unfinished_artifacts': [str(f.relative_to(ROOT)) for directory in [p, model_dir]
                for f in directory.rglob('*') if f.is_file() and
                (f.name in ['failure.json','logging_failure.json','.training.lock','.best.pt.tmp'] or f.suffix == '.tmp')],
            'model_artifact_inventory': {str(f.relative_to(model_dir)): {'bytes':f.stat().st_size,'sha256':sha(f)}
                                       for f in sorted(model_dir.rglob('*')) if f.is_file()}}
    report['runs'][row['experiment_id']] = item

    if row['model'] in ['LDA', 'CNN', 'CNN_LSTM']:
        def verify_single(p=p, run=run):
            result = {'partitions':{}, 'provenance_hashes':{}}
            split = rows(p / 'split_used.csv')
            for part in ['train', 'validation', 'test']:
                saved = read(p / (part + '_metrics.json'))
                pred = rows(p / (part + '_predictions.csv'))
                names = saved['class_names']
                truth = np.array([names.index(r['true_label']) for r in pred])
                prob = np.array([[float(r['probability_' + name]) for name in names] for r in pred])
                calculated = evaluate(truth, prob, names)
                compare(calculated, saved, part + '_metrics')
                compare(calculated, run['metrics'][part], part + '_run_metrics')
                expected = [r for r in split if r['split'] == part]
                compare([(r['sample_id'],r['path'],r['class_name']) for r in expected],
                        [(r['sample_id'],r['path'],r['true_label']) for r in pred], part + '_identity')
                assert all(r['predicted_label'] == names[int(np.argmax(pr))] for r,pr in zip(pred, prob))
                result['partitions'][part] = {'rows':len(pred),'all_saved_metrics_recomputed':True,'identities_and_labels_match':True}
            provenance = read(p / 'provenance.json')
            for name, digest in provenance['source_sha256'].items():
                assert sha(ROOT / name) == digest, ('source changed', name)
            for name,key in [('data/splits/fixed_split.csv','split_sha256'),('data/processed/inputs_v1.npz','cache_sha256')]:
                assert sha(ROOT / name) == provenance[key], ('data changed',name)
            result['provenance_hashes'] = {'source_files':len(provenance['source_sha256']),'split_and_cache':True}
            return result
        item['fresh_verification'] = checked(row['experiment_id'], verify_single)
    else:
        module = 'src.verify_ptbxl_spatiotemporal' if 'spatiotemporal' in row['experiment_id'] else 'src.verify_ptbxl_run'
        verifier = importlib.import_module(module).verify
        item['fresh_verification'] = checked(row['experiment_id'], lambda v=verifier, rid=row['experiment_id']: v(rid))


def hash_mapping(mapping, base=ROOT):
    matches, mismatches, missing = [], [], []
    for name, digest in mapping.items():
        if isinstance(digest, dict):
            digest = digest.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64:
            continue
        p = Path(name)
        if not p.is_absolute():
            p = base / name
        if not p.exists():
            missing.append(name)
        elif sha(p) == digest:
            matches.append(name)
        else:
            mismatches.append(name)
    return {'matches':len(matches),'mismatches':mismatches,'missing':missing}


manifest_paths = list((ROOT / 'experiments/logs').glob('*delivery_manifest.json'))
manifest_paths += list((ROOT / 'results').glob('*/*/figures/report_manifest.json'))
manifest_paths += [ROOT / 'results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.json']
manifest_paths += list((ROOT / 'results').glob('*/*/figure_review.json'))
for p in manifest_paths:
    obj = read(p)
    item = {}
    for key in ['files','artifacts_sha256','artifact_sha256','input_sha256','output_sha256',
                'source_artifact_sha256','implementation_sha256','output_artifact_sha256','figures','figure_sha256']:
        if isinstance(obj.get(key), dict):
            item[key] = hash_mapping(obj[key], p.parent if key == 'output_artifact_sha256' else ROOT)
    report['hash_manifests'][str(p.relative_to(ROOT))] = item

report['spatiotemporal_delivery_manifest_exists'] = any('spatiotemporal' in p.name for p in (ROOT / 'experiments/logs').glob('*delivery_manifest.json'))
report['limitations'] = [
    'Original baselines have test-evaluation policy records but no explicit inference-call counter; later three runs record one final test inference each.',
    'Saved bookkeeping cannot prove absence of unrecorded work outside these experiments.',
    'Fresh verifiers hash encoded cached images and inspect checkpoints without model inference; raw rendering and pixel normalization are not recomputed.',
    'Bootstrap point estimates and structure are checked, but resampling is not repeated.',
    'Historical delivery manifests can refer to superseded overview documentation and historical append-only experiment-log states; list differences without treating all as corrupt models.',
    'PTB-XL single-label accuracy and balanced-accuracy experiment-log fields are intentionally blank; per-label binary balanced accuracies are available.',
    'Wall-clock and elapsed timers differ during sleep and cannot support controlled architecture speed comparisons.'
]
(OUT / 'results_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print('DONE', OUT / 'results_audit.json', 'failures', len(report['failures']), flush=True)
