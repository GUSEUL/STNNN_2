#!/bin/bash
# Robustness Batch Test Shell Script
# Use three different types of data files (Ra variation, Ha variation, and other)
# for different Gaussian noise levels.

NOISE_LEVELS=("0.01" "0.02" "0.03" "0.05")
PERCENTS=("1" "2" "3" "5")
MATERIAL="EG"

for i in "${!NOISE_LEVELS[@]}"
do
    n=${NOISE_LEVELS[$i]}
    p=${PERCENTS[$i]}
    
    OUT_DIR="robustness_results_${p}percent_${MATERIAL}"
    echo -e "\n\033[0;36m>>> Starting Batch Inference: Noise ${p}% | Material: ${MATERIAL} -> ${OUT_DIR} <<<\033[0m"

    # Execute python script with arguments: noise_level, output_dir, material
    python inference_noise.py "$n" "$OUT_DIR" "$MATERIAL"

    if [ $? -ne 0 ]; then
        echo -e "\033[0;31mError occurred at ${p}% noise level with material ${MATERIAL}. Stopping.\033[0m"
        exit 1
    fi
done

echo -e "\n\033[0;32m[Done] All batch tests completed successfully.\033[0m"
