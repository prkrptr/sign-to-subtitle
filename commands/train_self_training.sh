GPU_ID=0
NUM_WORKERS=32

SAVE_DIR=inference_output

FEATURES_PATH=/mnt/lynx2/datasets/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22
GT_SUB_PATH=$SAVE_DIR/pseudo_label/subtitles_postprocessing
PR_SUB_PATH=/mnt/lynx2/datasets/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction

OMP_NUM_THREADS=1 \
python main.py \
--features_path $FEATURES_PATH \
--gt_sub_path $GT_SUB_PATH \
--pr_sub_path $PR_SUB_PATH \
--gpu_id $GPU_ID \
--batch_size 64 \
--n_workers $NUM_WORKERS \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--jitter_location \
--jitter_abs \
--jitter_loc_quantity 3. \
--load_words False \
--load_subtitles True \
--lr 5e-6 \
--save_path $SAVE_DIR/train_self_training \
--train_videos_txt 'data/bobsl_train_1658.txt' \
--val_videos_txt 'data/bobsl_val_32.txt' \
--test_videos_txt 'data/bobsl_test_250.txt' \
--n_epochs 8 \
--shuffle_getitem True \
--concatenate_prior True \
--min_sent_len_filter 0.5 \
--max_sent_len_filter 20 \
--shuffle_words_subs 0.5 \
--drop_words_subs 0.15 \
--resume $SAVE_DIR/train/checkpoints/model_last.pt \
\
--remove_stopwords False \
--preprocess_words True \
--remove_be True \
--remove_have True \
--finetune_bert True \
\
--expand_pr_step 20 \
\
--model gt_align_invtransformer_neg_rel \
--neg_lambda 1.0 \
--rel_lambda 1.0 \

# For debug
# --random_subset_data 5 \
