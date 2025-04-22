#!/usr/bin/env python3
from argparse import Namespace
import torch

from classifiers.FFBase import FFBase
from common import build_model

from torchaudio import load


def load_model_for_interactive():
    args: Namespace = Namespace()
    args.extractor = "XLSR_300M"
    args.classifier = "FFConcat3"
    
    args.processor = "MHFA"
    model_mhfa, _ = build_model(args)
    assert isinstance(model_mhfa, FFBase)
    model_mhfa.load_state_dict(torch.load("FFConcat3_MHFA_finetune_7.pt", map_location=torch.device('cpu'), weights_only=True))

    # args.processor = "AASIST"
    # model_aasist, _ = build_model(args)
    # assert isinstance(model_aasist, FFBase)
    # model_aasist.load_state_dict(torch.load("FF_AASIST_finetune_5.pt", map_location=torch.device('cpu'), weights_only=True))

    print("Models loaded successfully")
    # print(model_mhfa, model_aasist)
    return model_mhfa.eval() #, model_aasist.eval()

if __name__ == "__main__":
    model = load_model_for_interactive()
