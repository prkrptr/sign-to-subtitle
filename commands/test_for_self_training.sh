GPU_ID=0
NUM_WORKERS=32

SAVE_DIR=inference_output

FEATURES_PATH=/mnt/lynx2/datasets/bobsl/bobsl/features/i3d_c2281_16f_m8_-15_4_d0.8_-3_22
GT_SUB_PATH=/mnt/lynx2/datasets/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction
PR_SUB_PATH=/mnt/lynx2/datasets/bobsl/bobsl/subtitles/audio-aligned-heuristic-correction

for VIDEOS_TXT in 'data/bobsl_train_1658.txt' 'data/bobsl_val_32.txt'; do

OMP_NUM_THREADS=1 \
python main.py \
--features_path $FEATURES_PATH \
--gt_sub_path $GT_SUB_PATH \
--pr_sub_path $PR_SUB_PATH \
--gpu_id $GPU_ID \
--n_workers $NUM_WORKERS \
--batch_size 1 \
--pr_subs_delta_bias 2.7 \
--gt_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro True \
--resume $SAVE_DIR/finetune/checkpoints/model_best.pt \
--save_path $SAVE_DIR/pseudo_label \
--test_videos_txt $VIDEOS_TXT \
\
--remove_stopwords False \
--preprocess_words True \
--remove_be True \
--remove_have True \
\
--expand_pr_step 20 \

done

# For debug
# --random_subset_data 5 \
