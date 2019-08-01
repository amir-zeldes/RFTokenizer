# Test using heb.sm3 model
# To reproduce 2018 paper results, use Python 3.5, sklearn 0.19.0, RFTokenizer V0.9.0
echo "Tokenizing SPMRL data with current model"
python tokenize_rf.py -m heb data/SPMRL_test_plain.tab > data/SPMRL_test_pred.tab
python f_score_segs.py data/SPMRL_test_gold.tab pred/SPMRL_test_pred.tab

echo ""
echo "Tokenizing Wiki5K data with current model"
python tokenize_rf.py -m heb data/Wiki5K_test_plain.tab > pred/Wiki5K_test_pred.tab
python f_score_segs.py data/Wiki5K_test_gold.tab pred/Wiki5K_test_pred.tab
