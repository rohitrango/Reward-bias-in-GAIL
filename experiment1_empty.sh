env=empty
print_config=''
rollout_hint=EmptyPPO
model_name=EmptyGAIL

############################
## In case you forget: model_name: EmptyGAIL
## rollout_hint: EmptyPPO
## env: empty
############################

for seed in {1..3}
do
    # This is for BC
   #  python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}bc bc 2> logs/exp1_${model_name}bc_${seed}.txt
    # This is for positive reward
    python3 -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}positive_terminal_100 total_timesteps=1000000 use_terminal_state_disc 
done
