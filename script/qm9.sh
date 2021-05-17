main='scgcn_qm9'

echo "" > results/qm9/$main.log
for target in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
do
    echo 'processing:' $target >> results/qm9/$main.log
    python3 -u qm9/$main.py --target $target >> results/qm9/$main.log
done
