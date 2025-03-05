GPU_ID=0
NUM_WORKERS=32

SAVE_DIR=inference_output

FEATURES_PATH=/mnt/lynx2/datasets/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22
GT_SUB_PATH=/mnt/lynx2/datasets/bobsl/bobsl/subtitles/manually-aligned
PR_SUB_PATH=/mnt/lynx2/datasets/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction

OMP_NUM_THREADS=1 \
python main.py \
--features_path $FEATURES_PATH \
--gt_sub_path $GT_SUB_PATH \
--pr_sub_path $PR_SUB_PATH \
--gpu_id $GPU_ID \
--n_workers $NUM_WORKERS \
--batch_size 1 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro True \
--resume $SAVE_DIR/finetune_self_training/checkpoints/model_best.pt \
--save_path $SAVE_DIR/test_self_training \
\
--remove_stopwords False \
--preprocess_words True \
--remove_be True \
--remove_have True \
\
--expand_pr_step 20 \

# For debug
# --random_subset_data 5 \

# Computed over 2642663 frames, 20338 sentences - Frame-level accuracy: 77.22 F1@0.10: 81.39 F1@0.25: 75.03 F1@0.50: 63.81
