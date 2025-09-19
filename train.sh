set -eux

export NGPU=8
export LOG_RANK=0
export CONFIG_FILE="config/llama3_2_1b_peoples_speech.toml"

torchrun --nproc_per_node=${NGPU} --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.train --job.config_file ${CONFIG_FILE}
