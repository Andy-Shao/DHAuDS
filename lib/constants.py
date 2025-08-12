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

DYN_SNR_L1 = [5, .5, 6]
DYN_SNR_L2 = [5, .5, 7]
DYN_PSH_L1 = [4, 5]
DYN_PSH_L2 = [5, 7]
DYN_TST_L1 = [.07, .01, .1]
DYN_TST_L2 = [.08, .01, .12]
ENQ_NOISE_L2_LIST = ['CAFE', 'CAR', 'HOME', 'REVERB', 'STREET']
ENQ_NOISE_L1_LIST = ['CAFE', 'CAR', 'HOME']
END1_NOISE_L2_LIST = ['DKITCHEN', 'NFIELD', 'STRAFFIC', 'PRESTO', 'TCAR', 'OOFFICE']
END1_NOISE_L1_LIST = ['NFIELD', 'PRESTO', 'TCAR', 'OOFFICE']
END2_NOISE_L2_LIST = ['DLIVING', 'NRIVER', 'OHALLWAY', 'PSTATION', 'SPSQUARE', 'TMETRO']
END2_NOISE_L1_LIST = ['DLIVING', 'OHALLWAY', 'SPSQUARE', 'TMETRO']
ENSC_NOISE_L2_LIST = ['doing_the_dishes', 'exercise_bike', 'running_tap', 'white_noise', 'dude_miaowing', 'pink_noise']
ENSC_NOISE_L1_LIST = ['exercise_bike', 'running_tap', 'white_noise', 'pink_noise']