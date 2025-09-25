import os
import sys
import random
import argparse
import numpy as np
import torch
import tqdm

from misc.utils import TrainingParams, get_datetime
from models.losses.loss import make_losses
from datasets.dataset_utils import make_dataloaders
from eval.evaluate import evaluate, print_eval_stats
from training.train_utils import *


def do_train(params: TrainingParams):
    # ----- Model -----
    from models.ImLPR import ImLPR
    model = ImLPR(params.model_params)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_name = params.model_name
    print(f'Model name: {model_name}')
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)

    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum(p.nelement() for p in model.parameters())
        print(f'Number of model parameters: {n_params}')
    print(f'Model device: {device}')

    # ----- Data -----
    dataloaders = make_dataloaders(params)

    # ----- Loss(es) -----
    loss_fn = make_losses(params)

    # ----- Optionally load previous weights -----
    prev_path = params.load_weights
    if os.path.exists(prev_path):
        previous_model = torch.load(prev_path, map_location="cpu")
        if torch.cuda.device_count() <= 1:
            previous_model = {k[7:] if k.startswith("module.") else k: v for k, v in previous_model.items()}
        missing = model.load_state_dict(previous_model, strict=False)
        print("[load] missing:", getattr(missing, "missing_keys", None))
        print("[load] unexpected:", getattr(missing, "unexpected_keys", None))
    else:
        print(f"[info] previous weights not found: {prev_path}")

    # ----- Optimizer -----
    if params.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif params.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {params.optimizer}")

    optimizer = optimizer_fn(model.parameters(), lr=params.lr)


    # ----- Scheduler -----
    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == "CustomCosineWarmupScheduler":
            scheduler = CustomCosineWarmupScheduler(optimizer, warmup=params.warmup_epochs, max_iters=params.epochs, min_value=0.1)
        else:
            raise NotImplementedError(f'Unsupported LR scheduler: {params.scheduler}')

    # ----- Step function (single vs multi-stage) -----
    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = training_step
    else:
        train_step_fn = multistaged_training_step

    # ----- Book-keeping -----
    stats = {'train': [], 'eval': []}
    phases = ['train', 'val'] if 'val' in dataloaders else ['train']
    if 'val' in dataloaders:
        stats['val'] = []

    best_recall = 0.0
    intra_flg = False

    # ----- Epoch loop -----
    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        metrics = {'train': {}, 'val': {}}

        # Optional eval cadence
        if epoch % 5 == 0:
            intra_flg = (epoch % 1000 == 1)
            resuls_eval = evaluate(model, device, params, intra_flg=intra_flg)
            if resuls_eval.get('ave_recall', 0.0) > best_recall:
                best_recall = resuls_eval['ave_recall']
                torch.save(model.state_dict(), model_pathname + "_best.pth")
                print('Best model saved')

        for phase in phases:
            running_stats, count_batches = [], 0
            global_iter = iter(dataloaders[phase]) if phase in dataloaders and dataloaders[phase] is not None else None

            while True:
                count_batches += 1
                if params.debug and count_batches > 2:
                    break
                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    running_stats.append({'global': temp_stats})
                except StopIteration:
                    break

            # Aggregate epoch stats
            epoch_stats = {}
            if running_stats:
                for substep in running_stats[0]:
                    epoch_stats[substep] = {}
                    for key in running_stats[0][substep]:
                        temp = [e[substep][key] for e in running_stats]
                        if isinstance(temp[0], dict):
                            epoch_stats[substep][key] = {k: np.mean([t[k] for t in temp]) for k in temp[0]}
                        elif isinstance(temp[0], np.ndarray):
                            epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                        else:
                            epoch_stats[substep][key] = np.mean(temp)

                stats[phase].append(epoch_stats)
                print_stats(phase, epoch_stats)

                # For external logging (e.g., wandb)
                metrics[phase]['loss1'] = epoch_stats['global']['loss']
                if 'num_non_zero_triplets' in epoch_stats['global']:
                    metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']
                if 'positive_ranking' in epoch_stats['global']:
                    metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']
                if 'recall' in epoch_stats['global']:
                    metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]
                if 'ap' in epoch_stats['global']:
                    metrics[phase]['AP'] = epoch_stats['global']['ap']

        # Finalize epoch
        if scheduler is not None:
            scheduler.step()

        if params.save_freq > 0 and epoch % params.save_freq == 0:
            torch.save(model.state_dict(), model_pathname + f"_{epoch}.pth")

        if params.batch_expansion_th is not None and stats['train']:
            le_train_stats = stats['train'][-1]
            rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
            if rnz < params.batch_expansion_th:
                dataloaders['train'].batch_sampler.expand_batch()

    # Save final model
    final_model_path = model_pathname + '_final.pth'
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)

    # Final evaluation
    stats_eval = evaluate(model, device, params, intra_flg=intra_flg)
    print_eval_stats(stats_eval)
    print('.')


# -------------------------
# CLI helpers (combined)
# -------------------------

def _env_info():
    cuda = torch.cuda.is_available()
    gpus = torch.cuda.device_count()
    dev = torch.device("cuda" if cuda else "cpu")
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda if cuda else None,
        "gpu_count": gpus,
        "device": str(dev),
        "gpu_name_0": torch.cuda.get_device_name(0) if cuda and gpus > 0 else None,
    }


def _set_seed(seed: int, deterministic: bool = False):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _main():
    parser = argparse.ArgumentParser(description="Train ImLPR model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model-specific configuration file")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.set_defaults(debug=False)

    # optional niceties
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None)")
    parser.add_argument("--device", type=str, default=None, help='Force device, e.g. "cuda:0" or "cpu" (trainer auto-chooses if not set)')
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic kernels where possible")

    args = parser.parse_args()

    print(f"Training config path: {args.config}")
    print(f"Model config path: {args.model_config}")
    print(f"Debug mode: {args.debug}")

    env = _env_info()
    print(f"[env] torch={env['torch_version']}  cuda_available={env['cuda_available']}  "
          f"cuda_version={env['cuda_version']}  gpu_count={env['gpu_count']}")
    if env["gpu_name_0"]:
        print(f"[env] gpu0={env['gpu_name_0']}")
    print(f"[env] device(auto)={env['device']}")

    _set_seed(args.seed, deterministic=args.deterministic)
    if args.seed is not None:
        print(f"[env] seed={args.seed}  deterministic={args.deterministic}")

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if args.device is not None:
        print(f"[env] requested device={args.device} (trainer will choose automatically unless you wire this through)")

    try:
        do_train(params)
    except KeyboardInterrupt:
        print("\n[interrupt] Training interrupted by user.", file=sys.stderr)
    except Exception as e:
        print(f"[error] Unhandled exception: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    _main()
