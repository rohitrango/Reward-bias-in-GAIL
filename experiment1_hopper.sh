env=hopper
print_config=''
rollout_hint=HopperPPO
model_name=HopperGAIL

############################
## In case you forget: model_name: EmptyGAIL
## rollout_hint: EmptyPPO
## env: empty
############################

for seed in {1..3}
do
    # This is for BC
    #python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}bc bc 2> logs/exp1_${model_name}bc_${seed}.txt
    # This is for positive reward
    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}positive # 2> logs/exp1_${model_name}positive_${seed}.txt
done
