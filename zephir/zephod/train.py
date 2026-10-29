"""
Training ZephOD feature detection network

Usage:
    train_zephod -h | --help
    train_zephod -v | --version
    train_zephod --index=<index> [options]

Options:
    -h --help                           show this message and exit.
    -v --version                        show version information and exit.
    --index=<index>                     path to json index of data directories to use as training data.
    --arch=<arch>                       network: zephod (original) or unet3d. [default: zephod]
    --batch_size=<batch_size>           batch size in volumes. [default: 1]
    --channel=<channel>                 data channel to use as input.
    --cuda=<cuda>                       check if a CUDA-compatible GPU is available for  use. [default: True]
    --frames=<frames>                   training frames (python list), applied to every dataset; default all.
    --gamma=<gamma>                     gamma correction, should match the tracker's --gamma. [default: 1]
    --loss_func=<loss_func>             loss function to use. [default: BCE]
    --lr_init=<lr_init>                 learning rate for optimizer. [default: 1.0]
    --amp=<amp>                         use float16 autocast. [default: False]
    --model=<model>                     name of model being trained (saved in the zephod package folder).
    --out=<out>                         checkpoint path; overrides --model.
    --n_epoch=<n_epoch>                 number of epochs to train for. [default: 64]
    --n_trn=<n_trn>                     number of training samples per epoch. [default: 50]
    --n_val=<n_val>                     number of datasets held out for validation (multi-dataset index). [default: 5]
    --override=<override>               override existing checkpoint. [default: False]
    --pos_weight=<pos_weight>           BCE weight on positive voxels. [default: 1.0]
    --seed=<seed>                       random seed. [default: 0]
    --target_sigma=<target_sigma>       Gaussian target sigma in xy pixels (z uses half); 0 = single voxel. [default: 0]
    --val_frames=<val_frames>           validation frames (python list) for a single dataset; never trained on.
"""

from docopt import docopt
import random
import shutil
import torch.nn as nn
import torch.optim as optim

from ..__version__ import __version__
from ..utils.getters import *
from zephir.models.losses import corr_loss
from ..utils.utils import *
from inspect import getmembers, isfunction

from . import channels
from ..utils import utils
from .model import ZephOD
from .unet import UNet3D


def default_model_kwargs(arch, n_channels):
    if arch == 'unet3d':
        return {'n_channels_in': n_channels, 'n_channels_out': 1, 'init_nodes': 16, 'depth': 3}
    # z is neither padded nor pooled, so output z lines up with input z
    return {
        'n_channels_in': 5 * n_channels,
        'n_channels_out': 1,
        'init_nodes': 16,
        'kernel': (1, 3, 3),
        'padding': (0, 1, 1),
        'pool_kernel': (1, 2, 2),
    }


def load_frames(dataset, frames=None):
    """Return {t: annotations (N, 3)} for frames where every keypoint has a human annotation."""
    annotation = get_annotation_df(dataset)
    t_note = np.sort(np.unique(annotation['t_idx']).astype(int))
    nn_list = [len(get_annotation(annotation, t)[0]) for t in t_note]
    n_neuron, t_neuron = np.max(nn_list), t_note[np.argmax(nn_list)]
    worldline_id, _, _ = get_annotation(annotation, t_neuron)
    shape_t = get_metadata(dataset)['shape_t']

    out = {}
    for t in t_note:
        if t >= shape_t or (frames is not None and t not in frames):
            continue
        u, annot, _ = get_annotation(annotation, t)
        if not all(w in u for w in worldline_id):
            continue  # partly annotated frame: missing neurons would count as background
        u_idx = np.array([np.where(u == w)[0][-1] for w in worldline_id], dtype=int)
        out[int(t)] = annot[u_idx, ...]
    print(f'{dataset}: {len(out)} fully annotated frames ({n_neuron} keypoints): {sorted(out)}')
    return out


def make_target(annots, shape_zyx, sigma):
    """Max of one Gaussian per keypoint, peak 1; sigma=0 gives single voxels."""
    labels = np.zeros((1, *shape_zyx), np.float32)
    if sigma <= 0:
        for a in annots:
            x, y, z = get_pixel(a, shape_zyx)
            labels[0, z, y, x] = 1
        return labels
    s = np.array([sigma / 2, sigma, sigma])
    r = np.ceil(3 * s).astype(int)
    gz, gy, gx = [np.arange(-k, k + 1) for k in r]
    g = np.exp(-(gz[:, None, None] ** 2 / (2 * s[0] ** 2)
                 + gy[None, :, None] ** 2 / (2 * s[1] ** 2)
                 + gx[None, None, :] ** 2 / (2 * s[2] ** 2)))
    for a in annots:
        x, y, z = get_pixel(a, shape_zyx)
        lo = np.array([z, y, x]) - r
        hi = np.array([z, y, x]) + r + 1
        clo, chi = np.maximum(lo, 0), np.minimum(hi, shape_zyx)
        sl = tuple(slice(a_, b_) for a_, b_ in zip(clo, chi))
        gsl = tuple(slice(a_ - l_, b_ - l_) for a_, b_, l_ in zip(clo, chi, lo))
        labels[(0, *sl)] = np.maximum(labels[(0, *sl)], g[gsl])
    return labels


def preprocess(vol, arch):
    """Network input channels for one volume (C, Z, Y, X) in [0, 1]."""
    if arch == 'unet3d':
        return (vol / (vol.max(axis=(1, 2, 3), keepdims=True) + 1E-8)).astype(np.float32)
    # same channel stack as ZephOD.preprocess
    out = vol.astype(np.float32)
    for c in range(vol.shape[0]):
        out = np.append(out, np.stack(
            [f[1](vol[c]) for f in getmembers(channels, isfunction)
             if f not in getmembers(utils, isfunction)], axis=0), axis=0)
    for c in range(out.shape[0]):
        out[c] = out[c] / (np.max(out[c]) + 1E-8)
    return out.astype(np.float32)


def random_flip(x, y):
    # same flips as the original augmentation (y, x, both, or none)
    k = random.randint(-1, 2)
    axes = {-1: (-2, -1), 0: (-2,), 1: (-1,)}.get(k, ())
    for ax in axes:
        x, y = np.flip(x, ax), np.flip(y, ax)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def train_model(
        index=Path('.'),
        dev=torch.device('cpu'),
        channel=None,
        n_epoch=10,
        loss_func='BCE',
        lr_init=0.1,
        batch_size=1,
        n_trn=100,
        n_val=10,
        model_kwargs=None,
        state_dict=None,
        arch='zephod',
        frames=None,
        val_frames=None,
        gamma=1.0,
        target_sigma=0.0,
        pos_weight=1.0,
        amp=False):

    with open(index) as f:
        dataset_paths = [Path(p) for p in json.load(f).values()]

    # split: explicit frames inside one dataset, otherwise whole datasets
    if val_frames is not None:
        overlap = set(val_frames) & set(frames or [])
        assert not overlap, f'frames in both train and validation: {sorted(overlap)}'
        trn = [(d, load_frames(d, frames)) for d in dataset_paths]
        trn = [(d, {t: a for t, a in fr.items() if t not in val_frames}) for d, fr in trn]
        val = [(d, load_frames(d, val_frames)) for d in dataset_paths]
    else:
        n_val = max(min(n_val, len(dataset_paths) - 1), 0)
        all_ = [(d, load_frames(d, frames)) for d in dataset_paths]
        trn, val = all_[:len(all_) - n_val], all_[len(all_) - n_val:]
    trn = [(d, fr) for d, fr in trn if fr]
    val = [(d, fr) for d, fr in val if fr]
    if not trn:
        raise ValueError('no fully annotated training frames')
    print(f'train frames: {[(d.name, sorted(fr)) for d, fr in trn]}')
    print(f'val frames: {[(d.name, sorted(fr)) for d, fr in val]}')

    cache = {}

    def sample(d, t, annots, augment=True):
        if (d, t) not in cache:
            cache[(d, t)] = preprocess(get_data(d, t, g=gamma, c=channel), arch)
        x = cache[(d, t)]
        y = make_target(annots, x.shape[1:], target_sigma)
        return random_flip(x, y) if augment else (x, y)

    d0, fr0 = trn[0]
    t0 = next(iter(fr0))
    x0, _ = sample(d0, t0, fr0[t0], augment=False)
    n_raw = x0.shape[0] if arch == 'unet3d' else x0.shape[0] // 5
    if model_kwargs is None:
        model_kwargs = default_model_kwargs(arch, n_raw)
    model = (UNet3D(**model_kwargs) if arch == 'unet3d' else ZephOD(**model_kwargs)).to(dev)
    if state_dict is not None:
        print('Loading model from existing state_dict...')
        model.load_state_dict(state_dict)
    optimizer = optim.Adadelta(model.parameters(), lr=lr_init)
    dev_type = 'cuda' if str(dev).startswith('cuda') else 'cpu'
    scaler = torch.amp.GradScaler(dev_type, enabled=amp)
    if loss_func == 'BCE':
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
    else:
        loss_function = corr_loss

    def run_val():
        if not val:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            for d, fr in val:
                for t, a in fr.items():
                    x, y = sample(d, t, a, augment=False)
                    pred = model(to_tensor(x, n_dim=5, dev=dev))
                    losses.append(loss_function(pred, to_tensor(y, n_dim=5, dev=dev)).item())
        model.train()
        return float(np.mean(losses))

    print('\n\n\n******* BEGIN TRAINING *******\n')
    model.train()
    loss_list = []
    pbar = tqdm(range(n_epoch), desc='Training model', unit='epochs')
    for epoch in pbar:
        ep_loss = []
        for _ in range(n_trn):
            d, fr = random.choice(trn)
            xs, ys = [], []
            for _ in range(batch_size):
                t = random.choice(list(fr))
                x, y = sample(d, t, fr[t])
                xs.append(x)
                ys.append(y)
            input_tensor = torch.from_numpy(np.stack(xs)).to(dev)
            target_tensor = torch.from_numpy(np.stack(ys)).to(dev)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp):
                pred = model(input_tensor)
            loss = loss_function(pred.float(), target_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss.append(loss.item())
        val_loss = run_val()
        loss_list.append((float(np.mean(ep_loss)), val_loss))
        pbar.set_postfix(Loss=f'{loss_list[-1][0]:.4f}', Val=f'{val_loss:.4f}' if val_loss is not None else '-')

    checkpoint = {
        'arch': arch,
        'model_kwargs': model_kwargs,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        'loss_list': loss_list,
        'train_frames': {str(d): sorted(fr) for d, fr in trn},
        'val_frames': {str(d): sorted(fr) for d, fr in val},
        'gamma': gamma,
        'channel': channel,
    }
    return checkpoint


def main():
    args = docopt(__doc__, version=f'ZephOD {__version__}')
    print(args, '\n')

    seed = int(args['--seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and args['--cuda'] in ['True', 'Y', 'y']:
        print('\n*** GPU available!\n')
        dev = 'cuda'
    else:
        dev = 'cpu'

    if args['--out'] is not None:
        checkpoint_path = Path(args['--out'])
        name = checkpoint_path.stem
    else:
        name = args['--model'] or 'model'
        checkpoint_path = Path(__file__).parent / f'{name}.pt'

    override = args['--override'] in ['True', 'Y', 'y']
    checkpoint = {'model_kwargs': None, 'state_dict': None}
    if checkpoint_path.is_file():
        print('\nPrevious checkpoint available.')
        (checkpoint_path.parent / 'backup').mkdir(exist_ok=True)
        _now = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M")
        shutil.copy(checkpoint_path, checkpoint_path.parent / 'backup' / f'{name}_{_now}.pt')
        if not override:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

    checkpoint = train_model(
        index=Path(args['--index']),
        dev=dev,
        channel=int(args['--channel']) if args['--channel'] else None,
        n_epoch=int(args['--n_epoch']),
        loss_func=str(args['--loss_func']),
        lr_init=float(args['--lr_init']),
        batch_size=int(args['--batch_size']),
        n_trn=int(args['--n_trn']),
        n_val=int(args['--n_val']),
        model_kwargs=checkpoint['model_kwargs'],
        state_dict=checkpoint['state_dict'],
        arch=args['--arch'],
        frames=eval(args['--frames']) if args['--frames'] else None,
        val_frames=eval(args['--val_frames']) if args['--val_frames'] else None,
        gamma=float(args['--gamma']),
        target_sigma=float(args['--target_sigma']),
        pos_weight=float(args['--pos_weight']),
        amp=args['--amp'] in ['True', 'Y', 'y'],
    )

    print(f'\nSaving checkpoint to: {checkpoint_path.as_posix()}')
    torch.save(checkpoint, checkpoint_path)
    print('\n\n*** DONE!\n')


if __name__ == '__main__':
    main()
