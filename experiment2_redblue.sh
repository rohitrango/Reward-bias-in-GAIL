env=redblue
rollout_hint=RedBluePPO
model_name=RedBlueGAIL
print_config=''
total_timesteps=11000000

############################
## In case you forget: model_name: EmptyGAIL
## rollout_hint: EmptyPPO
## env: empty
############################

for seed in {2,4}
do
    # This is for negative reward (No traj and traj)
    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}negative negative_reward 2> logs/exp2_${model_name}negative_${seed}.txt &
    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}negativeNoTraj negative_reward notraj total_timesteps=${total_timesteps} 2> logs/exp2_${model_name}negativeNoTraj_${seed}.txt &
    # With trajectory
    # This is for positive reward (No traj - traj is already done in experiment 1)
    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}positiveNoTraj notraj total_timesteps=${total_timesteps} 2> logs/exp2_${model_name}positiveNoTraj_${seed}.txt
done
