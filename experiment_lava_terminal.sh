env=lava
print_config=''
rollout_hint=LavaPPO
model_name=MiniGrid-LavaCrossingS9N0-v0

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
#    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail \
#    seed=${seed} model_name=${model_name}negative total_timesteps=1100000 negative_reward

    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail \
    seed=${seed} model_name=${model_name}negative_terminal total_timesteps=1100000 negative_reward use_terminal_state_disc

done
