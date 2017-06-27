#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:reseterr:exclusive_process
#PBS -l mem=10gb

cd ~/group/lab/divyanshi/deep-genomics/dev

# train
python NN-mp-seqonly.py ~/group/lab/divyanshi/data/encode-dream_v2/within-celltype/CTCF/iPSC_seqonly/model.1/CTCF mod.seq.h5py > mod.seqonly.out 2>&1 
python  conv3L-testseqonly.py ~/group/lab/divyanshi/data/encode-dream_v2/within-celltype/CTCF/iPSC_seqonly/test/CTCF.test mod.seq.h5py > mod.seqonly.test.out 2>&1
