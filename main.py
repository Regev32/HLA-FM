"""
Foundation Model for HLAs — unified entry point.
Configure the run by editing the variables below, then: python main.py
"""
from pretrain.train import run as run_pretrain
from pretrain.evaluate import run as run_eval_pretrain
from finetune.multitask.train import run as run_finetune
from finetune.multitask.evaluate import run as run_eval_finetune
from finetune.binding.peptide_mhc import PeptideMHCTask
from finetune.binding.mhc_tcr import MHCTCRTask

TASKS = [PeptideMHCTask(), MHCTCRTask()]


def main():
    run_pretrain()
    run_eval_pretrain()
    run_finetune(TASKS)
    run_eval_finetune(TASKS)

if __name__ == "__main__":
    main()
