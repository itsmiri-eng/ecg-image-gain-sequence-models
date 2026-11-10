"""Synthetic saved-artifact tests; no real cohort or model inference."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src import verify_ptbxl_hybrid_vit as verifier
from src.training.ptbxl_hybrid_vit import DEFAULT_CONFIG
from src.evaluation.multilabel import evaluate_multilabel
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap

FIELDS = ['experiment_id', 'date', 'model', 'status', 'dataset_version', 'split_version',
          'preprocessing', 'random_seed', 'hyperparameters', 'accuracy', 'balanced_accuracy',
          'macro_f1', 'weighted_f1', 'notes', 'output_directory']
COUNTS = {'train': 32, 'validation': 4, 'test': 4}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class HybridViTVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run_id = 'new'
        self.out = self.root / 'results/ptbxl_hybrid_vit' / self.run_id
        self.model = self.root / 'models/ptbxl_hybrid_vit' / self.run_id
        self.out.mkdir(parents=True)
        self.model.mkdir(parents=True)
        self.timestamp = '2026-09-25T00:00:00+00:00'
        self.config = {
            'model': 'ptbxl_image_hybrid_vit', 'dataset_id': 'synthetic_images',
            'source_dataset_root': 'source', 'cache_root': 'cache', 'split_version': 'official',
            'class_names': verifier.CLASS_NAMES, 'input_shape': verifier.INPUT_SHAPE,
            'source_manifest_sha256': 'a' * 64, 'source_verification_sha256': 'b' * 64,
            'source_protocol_sha256': 'c' * 64, 'seed': 42, 'preprocessing': {'same_pixels': True},
            'loss': 'unweighted BCE', 'selection_rule': 'macro AUC, AP, negative BCE',
            'threshold': .5, 'threshold_policy': 'fixed', 'uncertainty': {'confidence_level': .95, 'resamples': 5, 'seed': 7},
            'training': {**DEFAULT_CONFIG, 'device':'auto'},
            'architecture': {'feature_sequence':[70,96], 'head_configuration':verifier.HEAD_CONFIGURATION, 'trainable_parameters':23881, 'pretraining':False, 'streaming_causality':False},
            'sequence_interpretation': 'offline image features', 'test_policy': 'one exploratory test after selection',
            'comparison_limit': 'conditional single-configuration image comparison',
        }
        self.old_config = copy.deepcopy(self.config)
        self.old_config['model'] = 'ptbxl_image_cnn_lstm'
        self.old_config['training'] = {k: v for k, v in self.config['training'].items()
                                      if k not in ['embed_dim','num_heads','num_layers','feedforward_dim']}
        self.old_config['training']['hidden_size'] = 32
        write_json(self.root / verifier.PRIOR_CONFIG_PATH, self.old_config)
        self.metrics = evaluate_multilabel(np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]),
                                          np.array([[.8, .2, .8, .2, .8], [.2, .8, .2, .8, .2]]), verifier.CLASS_NAMES)

    def log_fixture(self):
        old_rows = []
        for name in ['lda', 'cnn', 'mendeley_cnn_lstm', 'ptbxl_lstm', 'ptbxl_tcn']:
            old_rows.append({key: name if key == 'experiment_id' else 'completed' if key == 'status' else 'old'
                             for key in FIELDS})
            old_rows[-1]['output_directory'] = f'results/prior/{name}'
        prior_path = self.out / 'prior_experiment_log.csv'
        write_csv(prior_path, old_rows, FIELDS)
        log = self.root / 'experiments/experiment_log.csv'
        row = {key: '' for key in FIELDS}
        row.update(experiment_id=self.run_id, model='PTBXL_Image_HybridViT', status='completed',
                   date=self.timestamp, dataset_version=self.config['dataset_id'], split_version=self.config['split_version'],
                   random_seed='42', output_directory=f'results/ptbxl_hybrid_vit/{self.run_id}',
                   hyperparameters=json.dumps(self.config['training']), preprocessing=json.dumps(self.config['preprocessing']),
                   macro_f1=str(self.metrics['macro_f1']), weighted_f1=str(self.metrics['weighted_f1']))
        write_csv(log, old_rows + [row], FIELDS)
        return prior_path, log, old_rows, row

    def protection_fixture(self):
        prior_log, log, old_rows, row = self.log_fixture()
        snapshot=self.root/'experiments/protocols/cnn_lstm_image_v1/baseline_log_before.csv'
        write_csv(snapshot,old_rows[:2],FIELDS)
        baseline=self.root/'experiments/protocols/ptbxl_hybrid_vit_image_v1/baseline_experiment_log.csv'
        baseline.parent.mkdir(parents=True,exist_ok=True)
        baseline.write_bytes(prior_log.read_bytes())
        original=self.root/'original-model.bin';original.write_bytes(b'preserved')
        write_json(self.root/'experiments/reviews/2026-09-24/baseline_lock.json',{'files':{
            'original-model.bin':verifier._sha(original),'experiments/experiment_log.csv':verifier._sha(snapshot)}})
        source=self.root/'source'
        for name in verifier.SOURCE_FILES:
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('source '+name)
        protected={'original-model.bin':verifier._sha(original)}
        protected.update({path.relative_to(self.root).as_posix():verifier._sha(path) for path in source.rglob('*') if path.is_file()})
        for old in old_rows:
            directory=self.root/old['output_directory'];model=self.root/'models/prior'/old['experiment_id']
            model.mkdir(parents=True);(model/'best.pt').write_bytes(b'prior model')
            write_json(directory/'run.json',{'model_directory':model.relative_to(self.root).as_posix()})
            for directory in (directory,model):
                protected.update({path.relative_to(self.root).as_posix():verifier._sha(path) for path in directory.rglob('*') if path.is_file()})
        prior={'baseline_log_sha256':verifier._sha(baseline),'sha256':protected}
        write_json(self.root/verifier.PRIOR_PATH,prior)
        write_json(self.out/'protected_artifacts.json',protected)
        write_json(self.out/'preservation.json',{'protected_artifacts_checked':len(protected),'changed_artifacts':[],
            'prior_model_rows_preserved_as_exact_byte_prefix':True,'prior_log_sha256':verifier._sha(prior_log),
            'baseline_preservation':{'immutable_baseline_artifacts_checked':1,'changed_baseline_artifacts':[],
                'original_log_sha256':verifier._sha(snapshot),'original_log_byte_prefix_preserved':True,
                'original_log_rows_preserved':2,'appended_log_rows':4,'current_log_sha256':verifier._sha(log)}})
        return source,prior,protected

    def test_protocol_keeps_prior_task_and_budget_but_replaces_head(self):
        verifier._verify_protocol(self.config, self.root)
        for key, value in [('threshold', .4), ('preprocessing', {'changed': True}), ('seed', 43)]:
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier._verify_protocol(changed, self.root)
        changed = copy.deepcopy(self.config)
        changed['training']['patience'] = 5
        with self.assertRaises(ValueError):
            verifier._verify_protocol(changed, self.root)
        changed = copy.deepcopy(self.config)
        changed['architecture']['streaming_causality'] = True
        with self.assertRaisesRegex(ValueError, 'streaming_causality'):
            verifier._verify_protocol(changed, self.root)

    def test_five_log_rows_exact_prefix_one_current_append_and_later_runs(self):
        prior, log, old, row = self.log_fixture()
        result = verifier._verify_log(prior, log, self.run_id, self.config, {'started_utc': self.timestamp}, self.metrics)
        self.assertEqual(result['prior_log_rows'], 5)
        self.assertEqual(result['appended_log_rows'], 1)
        write_csv(log, old + [row, dict(row, experiment_id='future')], FIELDS)
        later = verifier._verify_log(prior, log, self.run_id, self.config, {'started_utc': self.timestamp}, self.metrics)
        self.assertEqual(later['completed_run_log_sha256'], result['completed_run_log_sha256'])
        self.assertEqual(later['appended_log_rows'], 2)
        # A reproduction can follow a legitimate earlier failure without rewriting it.
        failed=dict(row,experiment_id='prior_failed',status='failed')
        write_csv(prior,old+[failed],FIELDS)
        write_csv(log,old+[failed,row],FIELDS)
        result=verifier._verify_log(prior,log,self.run_id,self.config,{'started_utc':self.timestamp},self.metrics)
        self.assertEqual(result['prior_log_rows'],6)

    def test_wrong_model_duplicate_nonblank_accuracy_or_prior_mutation_fails(self):
        for mode in ['model', 'duplicate', 'accuracy', 'prefix', 'intervening']:
            prior, log, old, row = self.log_fixture()
            records = old + [dict(row)]
            if mode == 'model':
                records[-1]['model'] = 'PTBXL_Image_CNN_LSTM'
            elif mode == 'duplicate':
                records.append(dict(row))
            elif mode == 'accuracy':
                records[-1]['accuracy'] = '.99'
            elif mode == 'prefix':
                records[0]['notes'] = 'changed'
            else:
                records.insert(5, dict(row, experiment_id='intervening'))
            write_csv(log, records, FIELDS)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verifier._verify_log(prior, log, self.run_id, self.config, {'started_utc': self.timestamp}, self.metrics)

    def test_protection_requires_prior_models_results_and_dataset(self):
        source,prior,declared=self.protection_fixture()
        result=verifier._verify_protection(self.root,self.out,source)
        self.assertEqual(result['changed_artifacts'],[])
        for name in ['original-model.bin','models/prior/lda/best.pt','source/protocol.json']:
            path=self.root/name;original=path.read_bytes();path.write_bytes(b'changed')
            with self.subTest(name=name),self.assertRaises(ValueError):
                verifier._verify_protection(self.root,self.out,source)
            path.write_bytes(original)
        del declared['models/prior/lda/best.pt']
        write_json(self.out/'protected_artifacts.json',declared)
        with self.assertRaises(ValueError):verifier._verify_protection(self.root,self.out,source)

    def completed_fixture(self):
        # Small synthetic partitions; shared cache verifier is separately covered
        # by prior tests and is mocked in this orchestration-level fixture.
        for name in verifier.CODE_FILES + ['src/verify_ptbxl_run.py']:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('frozen synthetic source ' + name)
        source, prior, _ = self.protection_fixture()
        for name, key in [('classification_manifest.csv', 'source_manifest_sha256'),
                          ('verification.json', 'source_verification_sha256'), ('protocol.json', 'source_protocol_sha256')]:
            self.config[key] = verifier._sha(source / name)
            self.old_config[key] = self.config[key]
        write_json(self.root / verifier.PRIOR_CONFIG_PATH, self.old_config)
        self.config['prior_artifacts_sha256']=verifier._sha(self.root/verifier.PRIOR_PATH)
        write_json(self.root/verifier.CONFIG_PATH,self.config)
        write_json(self.out/'config.json',self.config)
        self.norm = {'fit_partition': 'train', 'mean': .1, 'std': .2}
        self.parts, self.all_metrics = {}, {}
        current_id = 0
        for split, size in COUNTS.items():
            rows, predictions = [], []
            y = np.asarray([[int((i + j) % 2) for j in range(5)] for i in range(size)], np.int8)
            prob = np.where(y, .8, .2)
            for i, (truth, probability) in enumerate(zip(y, prob)):
                current_id += 1
                row = {'ecg_id': str(current_id), 'patient_id': str(current_id),
                       'image_path': f'images/{split}/ecg_{current_id:05d}.png', 'split': split}
                prediction = {k: row[k] for k in ['ecg_id', 'patient_id', 'image_path']}
                for name, target, score in zip(verifier.CLASS_NAMES, truth, probability):
                    row['label_' + name] = str(bool(target))
                    prediction.update({f'truth_{name}': int(target), f'probability_{name}': float(score),
                                       f'prediction_{name}': int(score >= .5)})
                rows.append(row)
                predictions.append(prediction)
            self.parts[split] = rows
            m = evaluate_multilabel(y, prob, verifier.CLASS_NAMES)
            self.all_metrics[split] = m
            write_csv(self.out / f'{split}_predictions.csv', predictions)
            write_json(self.out / f'{split}_metrics.json', m)
            write_csv(self.out / f'{split}_per_label.csv', [{'label': name, **m['per_label'][name]} for name in verifier.CLASS_NAMES])
            (self.out / f'{split}_confusion_matrices.png').write_bytes(b'plot presence')
            if split == 'test':
                ci = patient_cluster_bootstrap(y, prob, [row['patient_id'] for row in rows], verifier.CLASS_NAMES,
                                               n_resamples=5, seed=7)
                write_json(self.out / 'test_patient_bootstrap.json', ci)
        write_csv(self.out / 'development_split_used.csv', self.parts['train'] + self.parts['validation'])
        self.metrics = self.all_metrics['test']
        self.log_fixture()  # Log metrics/config now match this completed fixture.
        baseline = verifier._json(self.out / 'preservation.json')
        baseline['baseline_preservation']['current_log_sha256'] = verifier._sha(self.root / 'experiments/experiment_log.csv')
        write_json(self.out / 'preservation.json', baseline)
        validation = self.all_metrics['validation']
        key = [validation['macro_roc_auc'], validation['macro_average_precision'], -validation['binary_cross_entropy']]
        training = {'model': 'LeadImageHybridViT', 'config': self.config['training'], 'input_shape': verifier.INPUT_SHAPE,
                    'class_names': verifier.CLASS_NAMES, 'normalization': self.norm, 'training_records': 32, 'validation_records': 4,
                    'class_counts_train': [16] * 5, 'test_data_accessed': False, 'class_weighting': 'none', 'augmentation': 'none',
                    'pretrained_weights': False, 'resume_supported': False, 'device': 'cpu',
                    'checkpoint_path': str(self.model / 'best.pt'), 'history_path': str(self.model / 'history.csv'),
                    'num_parameters': 23881, 'best_epoch': 1, 'epochs_run': 5, 'best_selection_key': key, 'stopped_early': True,
                    'best_validation_metrics': validation, 'selected_epoch_online_train_bce': .3,
                    'checkpoint_selection': 'synthetic lexicographic rule', 'sequence_interpretation': 'image features',
                    'sequence_shape':[70,96], 'head_configuration':verifier.HEAD_CONFIGURATION, **verifier.HEAD_CONFIGURATION,
                    'device_probe': {'requested':'auto','resolved':'cpu','available':{'cpu':True,'cuda':False,'mps':False},'synthetic_only':True,'ecg_data_accessed':False,'fallback_reasons':[],
                    'attempts':[{'device':'cpu','status':'passed','output_shape':[1,5],'finite_forward_backward':True,'synthetic_records':1}]}}
        self.checkpoint = {'format_version': 1, 'model_name': 'LeadImageHybridViT', 'class_names': verifier.CLASS_NAMES,
                          'input_shape': verifier.INPUT_SHAPE, 'config': self.config['training'], 'normalization': self.norm,
                          'training_records': 32, 'class_counts_train': [16] * 5, 'best_epoch': 1,
                          'validation_selection_key': key, 'best_validation_metrics': validation,
                          'selection_rule': training['checkpoint_selection'], 'sequence_interpretation': 'image features',
                          'head_configuration': verifier.HEAD_CONFIGURATION, 'resume_supported': False,
                          'state_dict': {k:torch.zeros(shape) for k,shape in verifier._expected_state_shapes().items()}}
        torch.save(self.checkpoint, self.model / 'best.pt')
        checkpoint_hash = verifier._sha(self.model / 'best.pt')
        write_json(self.model / 'metadata.json', training)
        write_json(self.out / 'training_metadata.json', training)
        write_json(self.model / 'config.json', self.config['training'])
        write_json(self.model / 'progress.json', {'status': 'complete', 'phase': 'complete', 'epochs_run': 5,
                                                 'best_epoch': 1, 'test_data_accessed': False})
        (self.model / 'architecture.txt').write_text('synthetic architecture metadata')
        (self.model / 'training_curve.png').write_bytes(b'plot presence')
        history = []
        for epoch in range(1, 6):
            history.append({'epoch': epoch, 'learning_rate': .001, 'train_online_bce': .3,
                            **{f'val_{k}': v for k, v in validation.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                            'checkpoint_improved': epoch == 1, 'epochs_without_selection_improvement': epoch - 1,
                            'max_gradient_norm_before_clip': 1, 'epoch_seconds': 1})
        write_csv(self.model / 'history.csv', history)
        cfg_hash = verifier._sha(self.root / verifier.CONFIG_PATH)
        sources = {name: verifier._sha(self.root / name) for name in verifier.CODE_FILES}
        provenance = {'configuration_sha256': cfg_hash, 'source_sha256': sources, 'normalization': self.norm,
                      'started_utc': self.timestamp, 'git_commit':None, 'selection_rule': self.config['selection_rule'],
                      'test_policy': self.config['test_policy'], 'prior_artifacts_sha256': self.config['prior_artifacts_sha256']}
        write_json(self.root/verifier.PRETRAINING_PATH,{'created_utc':self.timestamp,
            'configuration_sha256':cfg_hash,'source_sha256':sources,
            'ecg_training_started':False,'test_outcome_used_for_design':False,
            'evidence_sha256':{verifier.PRIOR_PATH:verifier._sha(self.root/verifier.PRIOR_PATH)}})
        provenance['pretraining_freeze_sha256']=verifier._sha(self.root/verifier.PRETRAINING_PATH)
        write_json(self.out/'provenance.json',provenance)
        for name in verifier.CODE_FILES:
            path = self.out / 'source_snapshot' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.root / name).read_bytes())
        write_json(self.out / 'selection_frozen.json', {'checkpoint_sha256': checkpoint_hash, 'configuration_sha256': cfg_hash,
                   'selection_rule': self.config['selection_rule'], 'threshold': .5, 'training': training,
                   'test_inference_started': False, 'frozen_utc': self.timestamp})
        write_json(self.out / 'checkpoint_replay.json', {'split': 'train', 'records': 32,
                   'maximum_absolute_probability_difference': 0., 'passed': True, 'test_replay_performed': False,
                   'checkpoint_sha256': checkpoint_hash,'device':'cpu'})
        write_json(self.out / 'test_inference_state.json', {'status': 'completed', 'checkpoint_sha256': checkpoint_hash,
                   'threshold': .5, 'inference_calls': 1, 'records': 4, 'started_utc': self.timestamp, 'completed_utc': self.timestamp})
        run = {'status': 'completed', 'experiment_id': self.run_id, 'started_utc': self.timestamp, 'completed_utc': self.timestamp,
               'total_runtime_seconds': 10, 'git_commit':None, 'model_directory': self.model.relative_to(self.root).as_posix(),
               'representation': self.config['sequence_interpretation'], 'comparison_limit': self.config['comparison_limit'],
               'metrics': self.all_metrics, 'training': training, 'checkpoint_sha256': checkpoint_hash, 'test_inference_calls': 1,
               **{key: self.config[key] for key in ['model', 'dataset_id', 'split_version', 'threshold', 'uncertainty']}}
        write_json(self.out / 'run.json', run)
        write_csv(self.out / 'run_summary.csv', [{'run_id': self.run_id, 'dataset': self.config['dataset_id'], 'task': 'five-label',
                   **{key: self.metrics[key] for key in verifier.CI_METRICS}}])

    def run_verify(self):
        with patch.object(verifier, 'COUNTS', COUNTS), patch.object(verifier, '_verify_cache', return_value=(self.parts, self.norm, {'synthetic':True,'full_cache':True})), patch.object(torch.nn.Module, '__call__', side_effect=AssertionError('No inference allowed')):
            return verifier.verify(self.run_id, root=self.root)

    def test_complete_saved_run_verified_without_writes_or_inference(self):
        self.completed_fixture()
        before = {str(p): verifier._sha(p) for p in self.root.rglob('*') if p.is_file()}
        report = self.run_verify()
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['model_inference_performed'])
        self.assertEqual(report['experiment_log']['prior_log_rows'], 5)
        self.assertEqual(report['checkpoint_replay']['records'], 32)
        self.assertEqual(report['prediction_records_verified'], COUNTS)
        self.assertIn('run.json', report['run_artifact_sha256'])
        self.assertIn('source_snapshot/src/common.py', report['run_artifact_sha256'])
        self.assertTrue(all(not Path(k).is_absolute() for k in report['run_artifact_sha256']))
        self.assertEqual(before, {str(p): verifier._sha(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse((self.out / 'verification.json').exists())

    def test_checkpoint_wrong_model_head_or_nonfinite_tensor_is_rejected(self):
        self.completed_fixture()
        for field, value in [('model_name', 'LeadImageCNNLSTM'), ('head_configuration', {}),
                             ('state_dict', {'bad': torch.full((23881,), float('nan'))})]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint[field] = value
            with self.subTest(field=field), patch.object(torch, 'load', return_value=checkpoint), self.assertRaises(ValueError):
                self.run_verify()
        checkpoint=copy.deepcopy(self.checkpoint)
        checkpoint['state_dict']['classifier.weight']=checkpoint['state_dict']['classifier.weight'].reshape(32,5)
        with patch.object(torch,'load',return_value=checkpoint),self.assertRaisesRegex(ValueError,'tensor shapes'):
            self.run_verify()

    def test_pretraining_freeze_evidence_time_and_device_are_verified(self):
        self.completed_fixture()
        path=self.root/verifier.PRETRAINING_PATH
        original=path.read_bytes()
        frozen=verifier._json(path)
        frozen['created_utc']='2030-01-01T00:00:00+00:00'
        write_json(path,frozen)
        provenance=verifier._json(self.out/'provenance.json')
        provenance['pretraining_freeze_sha256']=verifier._sha(path)
        write_json(self.out/'provenance.json',provenance)
        with self.assertRaisesRegex(ValueError,'postdates'):
            self.run_verify()
        path.write_bytes(original)
        provenance['pretraining_freeze_sha256']=verifier._sha(path)
        write_json(self.out/'provenance.json',provenance)
        replay=verifier._json(self.out/'checkpoint_replay.json')
        replay['device']='cuda'
        write_json(self.out/'checkpoint_replay.json',replay)
        with self.assertRaisesRegex(ValueError,'device'):
            self.run_verify()

    def test_new_verification_output_never_overwrites_or_enters_prior_runs(self):
        report={'status':'passed','run_id':self.run_id}
        target=self.out/'verification.json'
        verifier.write_verification(report,target,root=self.root)
        before=target.read_bytes()
        with self.assertRaises(FileExistsError):
            verifier.write_verification(report,target,root=self.root)
        self.assertEqual(target.read_bytes(),before)
        with self.assertRaises(ValueError):
            verifier.write_verification(report,self.root/'results/ptbxl_cnn_lstm/old/new.json',root=self.root)
        review=self.root/'experiments/reviews/new/verification.json'
        verifier.write_verification(report,review,root=self.root)
        self.assertEqual(verifier._json(review),report)

    def test_changed_training_source_is_rejected_without_inference(self):
        self.completed_fixture()
        (self.root/'src/training/ptbxl_hybrid_vit.py').write_text('changed after source freeze')
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
            self.run_verify()

    def test_saved_prediction_or_test_inference_bookkeeping_tampering_fails(self):
        self.completed_fixture()
        state_path = self.out / 'test_inference_state.json'
        state = verifier._json(state_path)
        changed = dict(state, inference_calls=2)
        write_json(state_path, changed)
        with self.assertRaisesRegex(ValueError, 'inference_calls'):
            self.run_verify()
        write_json(state_path, state)
        path = self.out / 'test_predictions.csv'
        fields, rows = verifier._csv(path)
        rows[0]['truth_NORM'] = '1'
        write_csv(path, rows, fields)
        with self.assertRaisesRegex(ValueError, 'truth NORM'):
            self.run_verify()

    def test_incomplete_run_and_invalid_id_are_refused_read_only(self):
        for name, value in [('run.json', {'status': 'running'}), ('config.json', {}), ('provenance.json', {})]:
            write_json(self.out / name, value)
        with self.assertRaisesRegex(ValueError, 'run status'):
            verifier.verify(self.run_id, root=self.root)
        with self.assertRaisesRegex(ValueError, 'Invalid run ID'):
            verifier.verify('../escape', root=self.root)
        self.assertFalse((self.out / 'verification.json').exists())


class CacheScopeTests(unittest.TestCase):
    def cache_fixture(self, root, leak_patient=False):
        root = root.resolve()
        cache,source=root/'cache',root/'source'
        source.mkdir();cache.mkdir()
        for name in ('protocol.json','generation_summary.json','verification.json','generation_state.json'):
            write_json(source/name,{})
        source_rows,cache_rows=[],[]
        index=0
        for split,fold in [('train',1),('validation',9),('test',10)]:
            for offset in range(2):
                index+=1
                labels=[label for j,label in enumerate(verifier.CLASS_NAMES) if (offset+j)%2]
                row={'ecg_id':str(index),'patient_id':'1' if leak_patient else str(index),
                     'official_fold':str(fold),'image_path':f'images/{split}/ecg_{index:05d}.png','split':split,
                     **{f'label_{label}':str(label in labels) for label in verifier.CLASS_NAMES},
                     'eligible_for_classification':'True','exclusion_reason':'','labels_json':json.dumps(labels)}
                source_rows.append(row)
                relative=f'cachedimages/{split}/ecg_{index:05d}.png'
                path=cache/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'encoded synthetic image')
                cache_rows.append({**row,'cache_path':relative,'cache_sha256':verifier._sha(path),'cache_pixel_sha256':'a'*64})
        write_csv(source/'classification_manifest.csv',source_rows)
        write_csv(cache/'cache_manifest.csv',cache_rows)
        source_hashes={name:verifier._sha(source/name) for name in
                       ('classification_manifest.csv','protocol.json','generation_summary.json','verification.json','generation_state.json')}
        binding={'source_root':str(source),'class_names':verifier.CLASS_NAMES,
                 'preprocessing':{'input_shape':verifier.INPUT_SHAPE},'code_sha256':'b'*64,
                 'source_files_sha256':source_hashes}
        write_json(cache/'protocol.json',{'binding':binding})
        norm={'fit_partition':'train','training_images':2,'training_pixels':2*12*120*1120,
              'mean':.1,'std':.2,'raw_std':.2,'std_floor':1e-6}
        metadata={'status':'complete','input_shape':verifier.INPUT_SHAPE,'class_names':verifier.CLASS_NAMES,
                  'counts':{'train':2,'validation':2,'test':2},'images':6,'binding':binding,
                  'preprocessing':binding['preprocessing'],'source_manifest_sha256':source_hashes['classification_manifest.csv'],
                  'cache_manifest_sha256':verifier._sha(cache/'cache_manifest.csv'),
                  'source_png_hashes_checked_before_decoding':6,'normalization':norm}
        write_json(cache/'metadata.json',metadata)
        cfg={'cache_root':'cache','source_dataset_root':'source','source_manifest_sha256':source_hashes['classification_manifest.csv']}
        provenance={'cache_sha256':{name:verifier._sha(cache/name) for name in ('protocol.json','metadata.json','cache_manifest.csv')},
                    'source_sha256':{'src/ptbxl_learning_data.py':'b'*64},'normalization':norm}
        return cfg,provenance,cache/cache_rows[0]['cache_path']

    def test_full_cache_hashes_bytes_fast_mode_only_checks_manifest_and_inventory(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(verifier,'COUNTS',{'train':2,'validation':2,'test':2}):
            root=Path(directory);cfg,provenance,image=self.cache_fixture(root)
            full=verifier._verify_cache(root,cfg,provenance,full_cache=True)[2]
            fast=verifier._verify_cache(root,cfg,provenance,full_cache=False)[2]
            self.assertEqual(full['encoded_images_hashed'],6)
            self.assertEqual(fast['encoded_images_hashed'],0)
            self.assertFalse(fast['full_cache'])
            image.write_bytes(b'changed image')
            verifier._verify_cache(root,cfg,provenance,full_cache=False)
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
                verifier._verify_cache(root,cfg,provenance,full_cache=True)
            image.unlink()
            with self.assertRaisesRegex(ValueError,'inventory'):
                verifier._verify_cache(root,cfg,provenance,full_cache=False)

    def test_fast_mode_still_rejects_patient_fold_leakage(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(verifier,'COUNTS',{'train':2,'validation':2,'test':2}):
            root=Path(directory);cfg,provenance,_=self.cache_fixture(root,leak_patient=True)
            with self.assertRaisesRegex(ValueError,'Patient crosses'):
                verifier._verify_cache(root,cfg,provenance,full_cache=False)


if __name__ == '__main__':
    unittest.main()
