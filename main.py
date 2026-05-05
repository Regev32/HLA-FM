"""
Foundation Model for HLAs — unified entry point.
Configure the run by editing the variables below, then: python main.py
"""
from pretrain.train import run as run_pretrain
from pretrain.evaluate import run as run_eval_pretrain
from finetune.peptide_mhc.train import run as run_finetune
from finetune.peptide_mhc.evaluate import run as run_eval_finetune


def main():
    run_pretrain()
    run_eval_pretrain()
    run_finetune()
    run_eval_finetune()

if __name__ == "__main__":
    main()