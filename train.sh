# This will create the heb.sm3 trained model
# To reproduce paper results, use Python 3.5, sklearn 0.19.0
python tokenize_rf.py -t -m heb -p 0 -c data/heb.conf -l data/heb.lex -f data/heb.frq data/SPMRL_train_gold.tab
