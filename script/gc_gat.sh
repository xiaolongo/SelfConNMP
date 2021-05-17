main='o_gat'
dataset='REDDIT-BINARY'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True

main='la_gat'
dataset='REDDIT-BINARY'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True

main='sc_gat'
dataset='REDDIT-BINARY'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True

main='dsc_gat'
dataset='REDDIT-BINARY'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    # echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True

main='o_gat'
dataset='REDDIT-MULTI-5K'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    # echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True

main='la_gat'
dataset='REDDIT-MULTI-5K'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    # echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True

main='sc_gat'
dataset='REDDIT-MULTI-5K'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    # echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True

main='dsc_gat'
dataset='REDDIT-MULTI-5K'
hidden=32
echo "" >results/gat/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    # echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gat/$main.log
done
python readout.py --log gat/$main.log --itertion 10 --max True
