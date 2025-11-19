from pickle import load
from config.config import load_opts, save_opts
import torch
import os
import random
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from multiprocessing import Pool

from train import trainer_dict
from data import dataset_dict
from models import model_dict
from misc.evaluate_sub_alignment import eval_subtitle_alignment
from misc.postprocessing_remove_intersections import postprocessing_remove_intersections


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def pp(test_file, opts):
    return postprocessing_remove_intersections(
        test_file,
        path_subtitles=opts.pr_sub_path,
        path_probabilities=opts.save_probs_folder,
        path_postpro_subs=opts.save_postpro_subs_folder
    )


def main(opts):
    # --- device setup ---
    device = torch.device(f'cuda:{opts.gpu_id}' if int(opts.gpu_id) >= 0 and torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    set_seed(42, device)

    # --- prepare dataset and dataloader ---
    if opts.test_only:
        assert opts.centre_window, 'Window should be fixed at test time'
        assert not opts.jitter_location, 'Do not jitter location of prior at test time'
        assert opts.jitter_width_secs == 0, 'Do not jitter location of prior at test time'
        assert opts.drop_feats == 0, 'Do not drop features at test time'
        assert opts.shuffle_feats == 0, 'Do not shuffle features at test time'
        assert opts.shuffle_words_subs == 0, 'Do not shuffle subtitle words at test time'
        assert opts.drop_words_subs == 0, 'Do not drop subtitle words at test time'

        dataset = dataset_dict[opts.dataset](mode='test', opts=opts)
        dataloader = DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=0  # set 0 to avoid repeated worker init prints
        )
    else:
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        dataset = dataset_dict[opts.dataset](mode='train', opts=opts)
        dataloader = DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=opts.n_workers,
            worker_init_fn=worker_init_fn,
        )

        dataset_val = dataset_dict[opts.dataset](mode='val', opts=opts)
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=opts.n_workers
        )

    if len(dataloader) == 0:
        print("Length of dataloader is 0")
        return

    # --- initialize model ---
    model = model_dict[opts.model](opts=opts, dataloader=dataloader)
    model.to(device)
    print("Model's state_dict loaded to device:", device)

    trainer = trainer_dict[opts.trainer](model, opts, device=device)

    if opts.resume:
        trainer.load_checkpoint(opts.resume)

    # --- train or test ---
    if not opts.test_only:
        save_opts(opts, os.path.join(opts.save_path, "args.txt"))

        # validation before training
        res_val, best_metric = trainer.train(dataloader_val, mode='val', epoch=-1)

        for epoch in range(opts.n_epochs):
            print(f"Epoch {epoch+1}/{opts.n_epochs}")
            res_tr, _ = trainer.train(dataloader, mode='train', epoch=epoch)

            # evaluate every save_every_n
            if epoch % opts.save_every_n == 0:
                res_val, val_metric = trainer.train(dataloader_val, mode='val', epoch=epoch)
                trainer.save_checkpoint("model_last.pt")
                if val_metric >= best_metric:
                    best_metric = val_metric
                    trainer.save_checkpoint("model_best.pt")
    else:
        # --- test only ---
        res_val, val_metric = trainer.train(dataloader, mode='test', epoch=0)

        if opts.save_vtt:
            test_files = open(opts.test_videos_txt, "r").read().splitlines()
            if opts.random_subset_data < len(test_files):
                random.seed(opts.random_subset_data_seed)
                test_files = random.sample(test_files, opts.random_subset_data)

            # check subtitle extension
            sub_ext = ''
            sample_file = os.path.join(opts.gt_sub_path, test_files[0])
            if os.path.exists(sample_file + '/signhd.vtt'):
                sub_ext = '/signhd.vtt'
            elif os.path.exists(sample_file + '.vtt'):
                sub_ext = '.vtt'

            if sub_ext:
                gt_anno_paths = [Path(os.path.join(opts.gt_sub_path, p + sub_ext)) for p in test_files]

                if opts.dtw_postpro:
                    with Pool(opts.n_workers) as pool:
                        pool.starmap(pp, [(f, opts) for f in test_files])

                    eval_subtitle_alignment(
                        pred_path_root=Path(opts.save_postpro_subs_folder),
                        gt_anno_path_root=Path(opts.gt_sub_path),
                        list_videos=test_files,
                        fps=25,
                    )


if __name__ == '__main__':
    opts = load_opts()
    main(opts)
