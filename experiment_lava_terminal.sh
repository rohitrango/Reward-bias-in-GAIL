env=empty
print_config=''
rollout_hint=PPO
model_name=MiniGrid-Empty-6x6-v2

############################
## In case you forget: model_name: EmptyGAIL
## rollout_hint: EmptyPPO
## env: empty
############################

for seed in {1,2,4}
do
    # This is for BC
    #python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}bc bc 2> logs/exp1_${model_name}bc_${seed}.txt &
    # This is for positive reward
    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail \
    seed=${seed} model_name=${model_name}negative_check total_timesteps=1100000 negative_reward

    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail \
    seed=${seed} model_name=${model_name}negative_terminal_check2 total_timesteps=1100000 negative_reward use_terminal_state_disc

done
