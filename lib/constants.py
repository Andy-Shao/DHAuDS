from enum import Enum

PROJECT_TITLE='DHAuDS'
TRAIN_TAG='Train'
TTA_TAG='TTA'

dataset_dic = {
    'SpeechCommandsV2': 'SC2',
    'SpeechCommandsV1': 'SC1',
    'VocalSound': 'VS',
    'CochlScene': 'CS',
    'AudioMNIST': 'AM',
    'ReefSet': 'RS',
    'UrbanSound8K': 'US8'
}

architecture_dic = {
    'AuT': 'AuT',
    'HuBERT': 'HuB',
    'AST': 'AST'
}

hubert_level_dic = {
    'base': 'B', 'large': 'L', 'x-large': 'XL'
}

DYN_SNR_L1 = [7, 1, 10]
DYN_SNR_L2 = [5, .5, 7]
DYN_PSH_L1 = [4, 5]
DYN_PSH_L2 = [5, 7]
DYN_TST_L1 = [.07, .01, .1]
DYN_TST_L2 = [.08, .01, .12]
ENQ_NOISE_LIST = ['CAFE', 'CAR', 'HOME', 'REVERB', 'STREET']
END1_NOISE_LIST = ['DKITCHEN', 'NFIELD', 'STRAFFIC', 'PRESTO', 'TCAR', 'OOFFICE']
END2_NOISE_LIST = ['DLIVING', 'NRIVER', 'OHALLWAY', 'PSTATION', 'SPSQUARE', 'TMETRO']
ENSC_NOISE_LIST = ['doing_the_dishes', 'exercise_bike', 'running_tap', 'white_noise', 'dude_miaowing', 'pink_noise']