DATA_FOLDER="/media/psf/WVerbatim/data/mevolve/prox/quantitative/recordings/vicon_03301_01-1"
MODELS_FOLDER="/media/psf/WVerbatim/data/mevolve/prox"
python3 prox/main.py --config cfg_files/SMPLifyD.yaml \
    --recording_dir "${DATA_FOLDER}" \
    --output_folder "${DATA_FOLDER}/output" \
    --vposer_ckpt "${MODELS_FOLDER}/models/vposer_v1_0/" \
    --part_segm_fn "${MODELS_FOLDER}/models/smplx_parts_segm.pkl" \
    --model_folder "${MODELS_FOLDER}/models" \
    --use_cuda=0 \
    --interpenetration=0 \
    --save_meshes="True" \
    --render_results=0 \




"${MODELS_FOLDER}/vposerDecoderWeights.npz"