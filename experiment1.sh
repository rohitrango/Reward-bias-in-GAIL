env=$1
print_config=''
rollout_hint=$2
model_name=$3

if [ $# -ne 3 ]; then
    echo 'experiment1.sh env rollout modelname'
    exit
fi

############################
## In case you forget: model_name: EmptyGAIL
## rollout_hint: EmptyPPO
## env: empty
############################

for seed in {1,2,4}
do
    # This is for BC
    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}bc bc 2> logs/exp1_${model_name}bc_${seed}.txt &
    # This is for positive reward
    python -m imitation.scripts.train_adversarial ${print_config} with ${env} rollout_hint=${rollout_hint} gail seed=${seed} model_name=${model_name}positive 2> logs/exp1_${model_name}positive_${seed}.txt
done
