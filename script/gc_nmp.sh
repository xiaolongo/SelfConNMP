main='sc_nmp'
dataset='PROTEINS'
hidden=32
echo "" >results/nmp/$main.log
echo 'Dataset:' $dataset
for index in 1 2 3 4 5 6 7 8 9 10; do
    echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/nmp/$main.log
done
python readout.py --log nmp/$main.log --itertion 10 --max True
