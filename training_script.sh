#!/bin/bash

#SBATCH --job-name=test_moes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-small
#SBATCH --gres=gpu:1
#SBATCH --output=train_outs/experts/out/%x.%j.out
#SBATCH --error=train_outs/experts/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=20010751@st.phenikaa-uni.edu.vn

python train.py --epsilon 1e-8
python train.py --epsilon 1e-7
python train.py --epsilon 1e-6
python train.py --epsilon 1e-5
python train.py --epsilon 1e-4
python train.py --epsilon 1e-3
python train.py --epsilon 1e-2