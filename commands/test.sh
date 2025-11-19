GPU_ID=-1
NUM_WORKERS=8

SAVE_DIR="inference_output"

FEATURES_PATH="C:/Users/karlw/Desktop/tei stuff/thesis/SAT_sign-to-subtitle/data/0902_output" 
GT_SUB_PATH="C:/Users/karlw/Desktop/tei stuff/thesis/SAT_sign-to-subtitle/data/vtt_0902"
PR_SUB_PATH="C:/Users/karlw/Desktop/tei stuff/thesis/SAT_sign-to-subtitle/data/vtt_0902"

OMP_NUM_THREADS=1 \
python main.py \
--features_path "$FEATURES_PATH" \
--gt_sub_path "$GT_SUB_PATH" \
--pr_sub_path "$PR_SUB_PATH" \
--gpu_id $GPU_ID \
--n_workers $NUM_WORKERS \
--batch_size 1 \
--pr_subs_delta_bias 2.7 \
--fixed_feat_len 20 \
--centre_window \
--test_only \
--save_vtt True \
--save_probs True \
--dtw_postpro False \
--resume "C:/Users/karlw/Desktop/tei stuff/thesis/SAT_sign-to-subtitle/model_best.pt" \
--save_path "$SAVE_DIR/test" \
--remove_stopwords False \
--preprocess_words True \
--remove_be True \
--remove_have True \
--expand_pr_step 20 \
--test_videos_txt "data/for_testing.txt"
