main='o_gcn'
dataset='IMDB-BINARY'
hidden=32
echo "" >results/gcn/$main.log
for index in 1 2 3 4 5 6 7 8 9 10; do
    echo 'processing:' $index
    python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
done
python readout.py --log gcn/$main.log --itertion 10 --max True

# main='la_gcn'
# dataset='IMDB-BINARY'
# hidden=32
# echo "" >results/gcn/$main.log
# for index in 1 2 3 4 5 6 7 8 9 10; do
#     echo 'processing:' $index
#     python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
# done
# python readout.py --log gcn/$main.log --itertion 10 --max True

# main='sc_gcn'
# dataset='IMDB-BINARY'
# hidden=32
# echo "" >results/gcn/$main.log
# for index in 1 2 3 4 5 6 7 8 9 10; do
#     echo 'processing:' $index
#     python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
# done
# python readout.py --log gcn/$main.log --itertion 10 --max True

# main='dsc_gcn'
# dataset='IMDB-BINARY'
# hidden=32
# echo "" >results/gcn/$main.log
# for index in 1 2 3 4 5 6 7 8 9 10; do
#     echo 'processing:' $index
#     python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
# done
# python readout.py --log gcn/$main.log --itertion 10 --max True

# main='o_gcn'
# dataset='IMDB-MULTI'
# hidden=32
# echo "" >results/gcn/$main.log
# for index in 1 2 3 4 5 6 7 8 9 10; do
#     echo 'processing:' $index
#     python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
# done
# python readout.py --log gcn/$main.log --itertion 10 --max True

# main='la_gcn'
# dataset='IMDB-MULTI'
# hidden=32
# echo "" >results/gcn/$main.log
# for index in 1 2 3 4 5 6 7 8 9 10; do
#     echo 'processing:' $index
#     python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
# done
# python readout.py --log gcn/$main.log --itertion 10 --max True

# main='sc_gcn'
# dataset='IMDB-MULTI'
# hidden=32
# echo "" >results/gcn/$main.log
# for index in 1 2 3 4 5 6 7 8 9 10; do
#     echo 'processing:' $index
#     python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
# done
# python readout.py --log gcn/$main.log --itertion 10 --max True

# main='dsc_gcn'
# dataset='IMDB-MULTI'
# hidden=32
# echo "" >results/gcn/$main.log
# for index in 1 2 3 4 5 6 7 8 9 10; do
#     echo 'processing:' $index
#     python -u $main.py --dataset $dataset --hidden $hidden --idx $index >>results/gcn/$main.log
# done
# python readout.py --log gcn/$main.log --itertion 10 --max True
