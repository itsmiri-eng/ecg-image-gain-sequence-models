"""Publication diagram from the three frozen PTB-XL model specifications."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.common import ROOT, read_json, sha256_file, utc_now, write_json


def plot(output_dir):
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError('Choose a new architecture-figure output directory')
    inputs = [
        'experiments/configs/ptbxl_cnn_lstm_image_v1.json',
        'experiments/configs/ptbxl_spatiotemporal_image_v1.json',
        'experiments/configs/ptbxl_hybrid_vit_image_v1.json',
        'models/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/metadata.json',
        'models/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/metadata.json',
        'src/plot_ptbxl_architectures.py',
    ]
    cfg = [read_json(ROOT/p) for p in inputs[:3]]
    for key in ('input_shape', 'class_names', 'preprocessing', 'split_version'):
        assert cfg[0][key] == cfg[1][key] == cfg[2][key]
    assert cfg[0]['training']['hidden_size'] == 32
    assert cfg[1]['training']['dilations'] == [1, 2]
    assert cfg[2]['architecture']['head_configuration']['token_count'] == 70
    counts = [read_json(ROOT/p)['num_parameters'] for p in inputs[3:5]] + [cfg[2]['architecture']['trainable_parameters']]
    fig, ax = plt.subplots(figsize=(14, 8.5), constrained_layout=True)
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 14), ylim=(0, 9)); ax.axis('off')
    def box(x, y, w, h, title, body, color='#eef3f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.12',
                     facecolor=color,edgecolor='#62758a',linewidth=1.3))
        ax.text(x+w/2,y+h-.25,title,ha='center',va='top',fontsize=13,weight='bold',color='#19354d')
        ax.text(x+w/2,y+h-.67,body,ha='center',va='top',fontsize=11.5,linespacing=1.45,color='#243746')
    def arrow(start,end):
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','lw':1.6,'color':'#62758a'})
    ax.text(7,8.88,'Compact PTB-XL image models: shared representation, different readouts',
            ha='center',va='top',fontsize=17,weight='bold',color='#19354d')
    box(.3,6.65,3.5,1.55,'Twelve aligned lead images','12 × 120 × 1120 pixels\nTraining-only normalization\nNo native waveform inputs')
    box(4.45,6.65,4.15,1.55,'Same CNN design, trained separately','Depthwise + pointwise stages\nChannel widths: 12 → 8 → 16 → 24\nGroupNorm and ReLU')
    box(9.25,6.65,4.4,1.55,'Shared spatial feature sequence','Adaptive pooling: 24 × 4 × 70\n70 positions × 96 features\nPosition is image width, not signal time')
    arrow((3.85,7.45),(4.4,7.45)); arrow((8.65,7.45),(9.2,7.45))
    headings=['CNN–LSTM','CNN + temporal convolutions','Hybrid CNN + Transformer']
    bodies=[
        'One LSTM layer, hidden width 32\nMean across 70 outputs\nDropout 0.2 → five logits\n\n'+f'{counts[0]:,} total trainable parameters',
        'Projection: 96 → 32\nTwo residual blocks; kernel 3\nTwo convolutions per block\nDilations 1 and 2; mean pooling\n'+f'{counts[1]:,} total trainable parameters',
        'Projection: 96 → 32 + learned positions\nTwo layers; four attention heads\nFeedforward width 64; global attention\nLayerNorm → mean → dropout → logits\n'+f'{counts[2]:,} total trainable parameters',
    ]
    xs=[.3,4.95,9.6]
    for x,title,body,color in zip(xs,headings,bodies,['#e7f1f8','#eaf3e9','#f3ecf8']):
        box(x,2.8,4.05,2.75,title,body,color)
    # Branch at one common representation point; each run trains its own weights.
    ax.plot([2.325,11.625],[6.05,6.05],color='#62758a',lw=1.4)
    ax.plot([11.45,11.45],[6.60,6.05],color='#62758a',lw=1.4)
    for x in xs: arrow((x+2.025,6.05),(x+2.025,5.6))
    for x in xs: arrow((x+2.025,2.75),(x+2.025,2.25))
    box(.3,1.15,13.35,1.05,'Five independent sigmoid probabilities: NORM · MI · STTC · CD · HYP',
        'Official patient folds 1–8 / 9 / 10; validation checkpoint selection; fixed decision threshold 0.5','#f5f7f9')
    ax.text(7,.68,'All inputs are clean rendered ECG pages. The 70 positions include calibration/margins.\n'
            'Shared encoder architecture does not mean shared fitted weights. Head capacity and dropout topology differ.',
            ha='center',va='center',fontsize=10.5,color='#455b6b',linespacing=1.4)
    out.mkdir(parents=True)
    for ext in ('png','svg'):
        fig.savefig(out/f'ptbxl_architectures.{ext}',dpi=180)
    plt.close(fig)
    write_json(out/'manifest.json',{'created_utc':utc_now(),'purpose':'Architecture schematic; no fitted performance shown',
        'inputs_sha256':{p:sha256_file(ROOT/p) for p in inputs},
        'outputs_sha256':{p.name:sha256_file(p) for p in sorted(out.glob('ptbxl_architectures.*'))}})
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(plot(args.output_dir))
