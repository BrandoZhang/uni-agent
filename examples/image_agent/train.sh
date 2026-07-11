#!/usr/bin/env bash
#
# GRPO training launcher for the multi-modality image-generation agent (classic
# UniAgentLoop path). This is a compact, single-node starting recipe -- it mirrors
# examples/agent_train/single_node_debug.sh; see that script and
# examples/search_agent/train_fully_async_128K.sh for the full megatron/cluster knobs.
#
# Requires a GPU node and a served-able VLM (e.g. Qwen2.5-VL). It trains the ORCHESTRATOR's
# TEXT/TOOL policy (which prompt to write, when to call generate_image, when to finish); the
# image generator (media-ai) stays frozen. GRPO's group-relative baseline is the "did this
# rollout do better or worse than its siblings" signal.
#
# IMPORTANT: on this classic UniAgentLoop path the generated PIXELS do NOT enter the loss
# (convert_to_agent_output emits multi_modal_data={}; the rollout uses a text tokenizer, not an
# image processor) -- the vision encoder gets no gradient. This is effectively LLM-style RL over
# a VLM checkpoint. For true pixels-in-the-loss VLM RL, use the gateway/framework path described
# in examples/image_agent/README.md (§5).
#
# Run from the repository root (so Ray packages both verl/ and uni_agent/):
#   bash examples/image_agent/train.sh
#
set -xeuo pipefail

project_name='Uni-Agent-Image-Agent'
exp_name='GRPO-Qwen2.5-VL-image-agent'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/uni-agent"}
# A vision-language model so the actor can (in the gateway path) see images; on this
# classic path it is trained on the text/tool-use tokens of the orchestration.
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-VL-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# Produced by: python examples/data_preprocess/image_agent.py --local_save_dir <dir>
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/image_agent/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/image_agent/test.parquet"}

RUNTIME_ENV=${RUNTIME_ENV:-"examples/image_agent/runtime_env.yaml"}
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"examples/image_agent/agent_config.yaml"}

# --- algorithm ---
adv_estimator=grpo               # group-relative baseline = "better or worse than siblings"
rollout_mode="async"             # decouple rollout from training (image gen is slow)
rollout_name="vllm"              # vllm or sglang

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
n_resp_per_prompt=8              # GRPO group size
train_prompt_mini_bsz=8
total_rollout_steps=2000
test_freq=10

NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
gen_tp=${gen_tp:-2}

ray job submit --no-wait --runtime-env "${RUNTIME_ENV}" \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG_PATH}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=image_agent \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.test_freq="${test_freq}"
