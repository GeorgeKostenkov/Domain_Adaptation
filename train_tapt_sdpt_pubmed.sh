python AdaptSum/src/tapt_pretraining.py -path=pubmed \
                                -optim=adamw \
                                -train_size=10000 \
                                 -visible_gpu=0 \
                                 -save_path=TAPT_save
                                 
python AdaptSum/src/sdpt_pretraining.py -data_path=arxiv_10000 \
                                -train_from=TAPT_save/pubmed \
                                -visible_gpu=0 \
                                -saving_path=after_TAPT_save \
                                -epoch=10000 \
                                -optim=adamw
                                 
python AdaptSum/src/sdpt_pretraining.py -data_path=pubmed_10000 \
                                -train_from=after_TAPT_save/arxiv_10000 \
                                -visible_gpu=0 \
                                -saving_path=after_all_save \
                                -epoch=10000 \
                                -optim=adamw