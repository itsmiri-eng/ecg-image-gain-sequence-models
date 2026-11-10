"""Synthetic tests only: no source ECG images, outcomes or frozen test access."""
from contextlib import redirect_stdout
import csv
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.ptbxl_cnn_lstm import LeadImageCNNLSTM
from src.models.ptbxl_hybrid_vit import LeadImageHybridViT, INPUT_SHAPE, CLASS_NAMES, HEAD_CONFIGURATION
from src.training.ptbxl_hybrid_vit import (
    _get_config, _check_batch, _normalization, load_checkpoint,
    train_model, resolve_device, selection_key,
)


class SyntheticDataset(Dataset):
    def __init__(self, images, labels, normalization, split):
        self.normalization = normalization
        self.split, self.normalize = split, True
        self.metadata = {"class_names": list(CLASS_NAMES)}
        self.images = torch.from_numpy((images - normalization["mean"]) / normalization["std"])
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.accesses = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        self.accesses += 1
        return self.images[index], self.labels[index]


def datasets():
    rng = np.random.default_rng(19)
    images = rng.uniform(0, .25, size=(4, *INPUT_SHAPE)).astype(np.float32)
    validation = rng.uniform(.7, 1., size=(4, *INPUT_SHAPE)).astype(np.float32)
    labels = np.array([[1,0,1,0,1], [0,1,0,1,0], [1,0,0,1,0], [0,1,1,0,1]], dtype=np.float32)
    norm = {"fit_partition":"train", "mean":float(images.mean(dtype=np.float64)),
            "std":float(images.std(dtype=np.float64)), "training_images":4}
    return SyntheticDataset(images,labels,norm,"train"), SyntheticDataset(validation,labels,norm,"validation"), norm


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identical_seeded_encoder_shapes_and_finite_gradients(self):
        torch.manual_seed(42)
        reference = LeadImageCNNLSTM()
        torch.manual_seed(42)
        model = LeadImageHybridViT()
        self.assertEqual(sum(p.numel() for p in model.parameters()),23881)
        self.assertEqual(str(model.features),str(reference.features))
        for name,value in reference.features.state_dict().items():
            torch.testing.assert_close(model.features.state_dict()[name],value,rtol=0,atol=0)
        images = torch.randn(2,*INPUT_SHAPE)
        self.assertEqual(tuple(model.features(images).shape),(2,24,15,70))
        self.assertEqual(tuple(model.extract_sequence(images).shape),(2,70,96))
        output = model(images)
        self.assertEqual(tuple(output.shape),(2,5))
        torch.nn.functional.binary_cross_entropy_with_logits(output,torch.zeros(2,5)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.zeros(2,12,48,1000))
        with self.assertRaises(ValueError):
            model.encode_tokens(torch.zeros(2,69,96))

    def test_layers_have_independent_initial_parameters_and_no_cls(self):
        model = LeadImageHybridViT()
        a,b=model.transformer_layers
        for name in ("self_attn.in_proj_weight","linear1.weight","linear2.weight"):
            pa,pb=dict(a.named_parameters())[name],dict(b.named_parameters())[name]
            self.assertNotEqual(pa.data_ptr(),pb.data_ptr())
            self.assertFalse(torch.equal(pa,pb))
        self.assertEqual(tuple(model.position_embedding.shape),(1,70,32))
        self.assertTrue(all(layer.norm_first and layer.self_attn.batch_first for layer in model.transformer_layers))
        self.assertFalse(any('cls' in name for name,_ in model.named_parameters()))
        self.assertEqual(HEAD_CONFIGURATION['token_count'],70)

    def test_global_attention_reads_future_tokens_and_positions_break_permutation_symmetry(self):
        torch.manual_seed(5)
        model=LeadImageHybridViT(dropout=0).eval()
        sequence=torch.randn(1,70,96,requires_grad=True)
        tokens=model.encode_tokens(sequence)
        tokens[0,0,0].backward()
        self.assertGreater(float(sequence.grad[0,-1].abs().sum()),0)
        with torch.no_grad():
            original=model.encode_tokens(sequence.detach()).mean(1)
            reversed_order=model.encode_tokens(sequence.detach().flip(1)).mean(1)
            self.assertGreater(float((original-reversed_order).abs().max()),1e-6)
            model.position_embedding.zero_()
            original=model.encode_tokens(sequence.detach()).mean(1)
            reversed_order=model.encode_tokens(sequence.detach().flip(1)).mean(1)
            torch.testing.assert_close(original,reversed_order,rtol=1e-5,atol=1e-6)

    def test_configuration_targets_and_normalization_guards(self):
        for bad in ({"hidden_size":32},{"embed_dim":64},{"num_heads":2},{"num_layers":1},
                    {"feedforward_dim":128},{"device":"tpu"},{"num_workers":False},{"dropout":True},
                    {"seed":True},{"learning_rate":0},{"eta_min":.1}):
            with self.assertRaises(ValueError):
                _get_config(bad)
        for bad in (None,{}, {"fit_partition":"test","mean":0.,"std":1.},
                    {"fit_partition":"train","mean":False,"std":1.},
                    {"fit_partition":"train","mean":0.,"std":0.}):
            with self.assertRaises(ValueError):
                _normalization(bad)
        x,y=torch.zeros(1,*INPUT_SHAPE),torch.zeros(1,5)
        for bad_x,bad_y in [(x.double(),y),(x,y+.2),(x,y[:,:4])]:
            with self.assertRaises(ValueError):
                _check_batch(bad_x,bad_y)
        self.assertNotIn('test',inspect.signature(train_model).parameters)

    def test_selection_order_and_exact_ties(self):
        def key(auc,ap,bce):
            return selection_key({'macro_roc_auc':auc,'macro_average_precision':ap,'binary_cross_entropy':bce})
        self.assertGreater(key(.8,.1,3),key(.7,.9,.1))
        self.assertGreater(key(.8,.7,3),key(.8,.6,.1))
        self.assertGreater(key(.8,.7,.2),key(.8,.7,.3))
        self.assertFalse(key(.8,.7,.2)>key(.8,.7,.2))
        with self.assertRaises(ValueError):
            key(None,.7,.2)


class DeviceTests(unittest.TestCase):
    def test_cpu_probe_runs_actual_forward_backward(self):
        device,report=resolve_device({'device':'cpu','num_threads':2})
        self.assertEqual(str(device),'cpu')
        self.assertTrue(report['attempts'][0]['finite_forward_backward'])
        self.assertFalse(report['ecg_data_accessed'])

    def test_auto_records_unavailable_and_failed_candidates_without_silent_explicit_fallback(self):
        with patch('torch.cuda.is_available',return_value=False), patch('torch.backends.mps.is_available',return_value=False):
            device,report=resolve_device({'device':'auto','num_threads':2})
            self.assertEqual(str(device),'cpu')
            self.assertEqual([r['status'] for r in report['attempts']],['unavailable','unavailable','passed'])
            for name in ('cuda','mps'):
                with self.assertRaises(RuntimeError):
                    resolve_device({'device':name})
        def probe(device,cfg):
            if device=='cuda':
                raise RuntimeError('synthetic unsupported operation')
            return {'finite_forward_backward':True}
        with patch('torch.cuda.is_available',return_value=True), patch('torch.backends.mps.is_available',return_value=False), patch(
                'src.training.ptbxl_hybrid_vit._probe_device',side_effect=probe), patch('torch.cuda.manual_seed_all'):
            device,report=resolve_device({'device':'auto','num_threads':2})
            self.assertEqual(str(device),'cpu')
            self.assertEqual(report['attempts'][0]['status'],'failed')
            with self.assertRaises(RuntimeError):
                resolve_device({'device':'cuda','num_threads':2})


class TrainingTests(unittest.TestCase):
    cfg={'max_epochs':1,'patience':1,'batch_size':2,'num_threads':2}

    def test_deterministic_training_checkpoint_replay_progress_and_no_overwrite(self):
        train,val,norm=datasets()
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            first_dir,second_dir=Path(directory)/'first',Path(directory)/'second'
            first,metadata=train_model(train,val,self.cfg,first_dir,norm,[2]*5)
            self.assertEqual(train.accesses,len(train))
            self.assertEqual(val.accesses,len(val))
            second,_=train_model(train,val,self.cfg,second_dir,norm,[2]*5)
            probabilities=first.predict_proba(val)
            np.testing.assert_array_equal(probabilities,second.predict_proba(val))
            np.testing.assert_array_equal(probabilities,load_checkpoint(first_dir/'best.pt').predict_proba(val))
            np.testing.assert_array_equal(probabilities,load_checkpoint(first_dir/'best.pt',device=first.device).predict_proba(val))
            self.assertFalse(np.allclose(probabilities.sum(1),1.))
            self.assertEqual(metadata['num_parameters'],23881)
            self.assertEqual(metadata['normalization'],norm)
            self.assertEqual(metadata['head_configuration'],HEAD_CONFIGURATION)
            self.assertFalse(metadata['test_data_accessed'])
            self.assertEqual(metadata['device_probe']['resolved'],'cpu')
            progress=json.loads((first_dir/'progress.json').read_text())
            self.assertEqual(progress['status'],'complete')
            with self.assertRaises(FileExistsError):
                train_model(train,val,self.cfg,first_dir,norm,[2]*5)
            checkpoint=torch.load(first_dir/'best.pt',weights_only=True)
            for mutation in ('model','config','head','state'):
                changed=dict(checkpoint)
                if mutation=='model': changed['model_name']='LeadImageCNNLSTM'
                elif mutation=='config': changed['config']={**changed['config'],'num_heads':2}
                elif mutation=='head': changed['head_configuration']={}
                else:
                    changed['state_dict']={k:v.clone() for k,v in changed['state_dict'].items()}
                    changed['state_dict']['position_embedding'][0,0,0]=float('nan')
                path=Path(directory)/f'wrong_{mutation}.pt';torch.save(changed,path)
                with self.assertRaises(ValueError):load_checkpoint(path)

    def test_validation_only_selection_earliest_tie_and_early_stopping(self):
        train,val,norm=datasets()
        first={'macro_roc_auc':.7,'macro_average_precision':.5,'binary_cross_entropy':.3}
        second={'macro_roc_auc':.7,'macro_average_precision':.6,'binary_cross_entropy':.8}
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()),patch(
                'src.training.ptbxl_hybrid_vit._evaluate_validation',side_effect=[first,second,second]):
            _,metadata=train_model(train,val,{**self.cfg,'max_epochs':5},directory,norm,[2]*5)
            self.assertEqual(metadata['best_epoch'],2)
            self.assertEqual(metadata['epochs_run'],3)
            self.assertTrue(metadata['stopped_early'])
            with (Path(directory)/'history.csv').open() as f: history=list(csv.DictReader(f))
            self.assertEqual([r['checkpoint_improved'] for r in history],['True','True','False'])
            self.assertGreater(float(history[0]['learning_rate']),float(history[1]['learning_rate']))

    def test_test_split_and_wrong_counts_rejected_without_completion(self):
        train,val,norm=datasets()
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            val.split='test'
            with self.assertRaises(ValueError):train_model(train,val,self.cfg,Path(directory)/'bad_split',norm,[2]*5)
            self.assertEqual(train.accesses,0)
            val.split='validation'
            with self.assertRaises(ValueError):train_model(train,val,self.cfg,Path(directory)/'bad_count',norm,[1]*5)
            progress=json.loads((Path(directory)/'bad_count/progress.json').read_text())
            self.assertEqual(progress['status'],'failed')
            self.assertFalse((Path(directory)/'bad_count/best.pt').exists())


if __name__=='__main__':
    unittest.main()
