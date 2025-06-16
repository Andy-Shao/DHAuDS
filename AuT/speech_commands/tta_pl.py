import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms

from lib.utils import print_argparse, make_unless_exits
from lib import constants
from lib.spdataset import SpeechCommandsV1, SpeechCommandsV2
from lib.dataset import mlt_load_from, mlt_store_to, MultiTFDataset, IdxSet
from lib.component import Components, AudioPadding, BackgroundNoiseByFunc, DoNothing, time_shift
from lib.component import AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from AuT.speech_commands.analysis import noise_source, load_weight, inference
from AuT.speech_commands.train import build_model, lr_scheduler
from AuT.speech_commands.tta import build_optimizer, nucnm
from AuT.lib.plr import plr
from AuT.lib.loss import SoftCrossEntropyLoss

def plloss(args:argparse, mem_label, idxes, outputs) -> torch.Tensor:
    from torch.nn import functional as F
    if args.cls_rate > 0.:
        with torch.no_grad():
            pred = mem_label[idxes]
        if args.cls_mode == 'logsoft_ce':
            classifier_loss = SoftCrossEntropyLoss(outputs, pred)
            classifier_loss = torch.mean(classifier_loss)
        elif args.cls_mode == 'soft_ce':
            # softmax_output = nn.Softmax(dim=1)(outputs[0:batch_size])
            softmax_output = F.softmax(outputs, dim=1)
            classifier_loss = nn.CrossEntropyLoss()(softmax_output, pred)
        elif args.cls_mode == 'logsoft_nll':
            # softmax_output = nn.LogSoftmax(dim=1)(outputs[0:batch_size])
            softmax_output = F.log_softmax(outputs, dim=1)
            _, pred = torch.max(pred, dim=1)
            classifier_loss = nn.NLLLoss(reduction='mean')(softmax_output, pred)
        classifier_loss = args.cls_rate * classifier_loss
    else:
        classifier_loss = torch.tensor(.0).to(args.device)
    return classifier_loss

def obtain_label(data_loader:DataLoader, auT:nn.Module, auC:nn.Module, args:argparse.Namespace) -> tuple:
    from torch.nn import functional as F
    auT.eval(); auC.eval()
    # Accumulate feat, logint and gt labels
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = inputs.to(args.device)
            outputs, features = auC(auT(inputs)[0])
            if idx == 0:
                all_feature = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_feature = torch.cat([all_feature, features.float().cpu()], dim=0)
                all_output = torch.cat([all_output, outputs.float().cpu()], dim=0)
                all_label = torch.cat([all_label, labels.float()], dim=0)
        inputs = None
        features = None
        outputs = None
        ##################### Done ##################################
        # Clustering
        all_output = F.softmax(all_output, dim=1)
        _, predict = torch.max(all_output, dim=1)

        mean_all_output = torch.mean(all_output, dim=0).numpy()

        # find centroid per class
        if args.distance == 'cosine':
            ######### Not Clear (looks like feature normalization though)#######
            all_feature = torch.cat((all_feature, torch.ones(all_feature.size(0), 1)), dim=1)
            all_feature = (all_feature.t() / torch.norm(all_feature, p=2, dim=1)).t() # here is L2 norm
            ### all_fea: extractor feature [bs,N]. all_feature is g_t in paper
        all_feature = all_feature.float().cpu().numpy()
        K = all_output.size(1) # number of classes
        aff = all_output.float().cpu().numpy() ### aff: softmax output [bs,c]

        # got the initial normalized centroid (k*(d+1))
        initc = aff.transpose().dot(all_feature)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        cls_count = np.eye(K)[predict].sum(axis=0) # total number of prediction per class
        labelset = np.where(cls_count >= args.threshold) ### index of classes for which same sampeled have been detected # returns tuple
        labelset = labelset[0] # index of classes for which samples per class greater than threshold
        # labelset == [0, 1, 2, ..., 29]

        # dd is the data distance between data and central point.
        dd = all_feature @ initc[labelset].T # <g_t, initc>
        dd = np.exp(dd) # amplify difference
        pred_label = dd.argmax(axis=1) # predicted class based on the minimum distance
        pred_label = labelset[pred_label] # this will be the actual class

        for round in range(args.initc_num): # calculate initc and pseduo label multi-times
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_feature)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = all_feature @ initc[labelset].T
            dd = np.exp(dd)
            pred_label = dd.argmax(axis=1)
            pred_label = labelset[pred_label]
        
        dd = nn.functional.softmax(torch.from_numpy(dd), dim=1)
        return pred_label, dd.numpy().astype(np.float32), mean_all_output,

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2', 'SpeechCommandsV1'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
    ap.add_argument('--vocalsound_path', type=str)
    ap.add_argument('--cochlscene_path', type=str)
    ap.add_argument('--corruption_level', type=float)
    ap.add_argument('--corruption_type', type=str, choices=['doing_the_dishes', 'exercise_bike', 'running_tap', 'VocalSound', 'CochlScene'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--auT_lr_decay', type=float, default=1.)
    ap.add_argument('--auC_lr_decay', type=float, default=1.)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--origin_auT_weight', type=str)
    ap.add_argument('--origin_cls_weight', type=str)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    #pl args
    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--initc_num', type=int, default=1)
    ap.add_argument('--plr', action='store_true')
    ap.add_argument('--cls_rate', type=float, default=0.2, help='lambda 2 | Pseudo-label loss capable')
    ap.add_argument('--cls_mode', type=str, default='soft_ce', choices=['logsoft_ce', 'soft_ce', 'logsoft_nll'])
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    ap.add_argument('--alpha', type=float, default=0.9)

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
    elif args.dataset == 'SpeechCommandsV1':
        args.class_num = 30
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = 'AuT'
    args.output_path = os.path.join(args.output_path, args.dataset, arch, 'TTA')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{arch}-{constants.dataset_dic[args.dataset]}-{constants.corruption_dic[args.corruption_type]}-{args.corruption_level}-pl', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
    if args.dataset == 'SpeechCommandsV2':
        SpeechCommandsV2(root_path=args.dataset_root_path, mode='testing', download=True)
        corrupted_set = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode='testing', download=False, 
            data_tf=Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=args.corruption_type), is_random=True),
            ])
        )
    elif args.dataset == 'SpeechCommandsV1':
        corrupted_set = SpeechCommandsV1(
            root_path=args.dataset_root_path, mode='test',  include_rate=False,
            data_tfs=Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                BackgroundNoiseByFunc(noise_level=args.corruption_level, noise_func=noise_source(args, source_type=args.corruption_type), is_random=True)
            ])
        )
    dataset_root_path = os.path.join(args.cache_path, args.dataset, args.corruption_type, str(args.corruption_level))
    index_file_name = 'metaInfo.csv'
    mlt_store_to(
        dataset=corrupted_set, 
        index_file_name=index_file_name, root_path=dataset_root_path,
        data_tfs=[DoNothing()]
    )

    corrupted_set = mlt_load_from(
        root_path=dataset_root_path, 
        index_file_name=index_file_name,
    )
    corrupted_set = MultiTFDataset(
        dataset=corrupted_set,
        tfs=[
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=True, is_bidirection=False),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 104
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 104
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
        ]
    )
    corrupted_set = IdxSet(dataset=corrupted_set)

    corrupted_loader = DataLoader(dataset=corrupted_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    test_set = mlt_load_from(
        root_path=dataset_root_path, index_file_name=index_file_name,
        data_tfs=[
            Components(transforms=[
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 104
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        ]
    )
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    auTmodel, clsmodel = build_model(args)
    load_weight(args=args, mode='origin', auT=auTmodel, auC=clsmodel)
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel)

    print('Pre-adaptation')
    accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
    print(f'Accurayc is: {accuracy:.4f}%, sample size is: {len(corrupted_set)}')

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Scanning...')
        mem_label, dd, mean_all_output = obtain_label(data_loader=test_loader, auT=auTmodel, auC=clsmodel, args=args)

        if args.plr:
            if epoch == 0:
                prev_mem_label = mem_label
                mem_label = dd
            else:
                mem_label = plr(prev_mem_label, mem_label, dd, args.class_num, alpha = args.alpha)
                prev_mem_label = mem_label.argmax(axis=1).astype(int)
        else:
            mem_label = dd
        # print('Completed finding Pseudo Labels\n')
        mem_label = torch.from_numpy(mem_label).to(args.device)
        dd = torch.from_numpy(dd).to(args.device)
        mean_all_output = torch.from_numpy(mean_all_output).to(args.device)

        print('Adaptating...')
        auTmodel.train(); clsmodel.train()
        ttl_size = 0.; ttl_loss = 0.; ttl_nucnm_loss = 0.; ttl_pl_loss = 0.
        for idxes, fs1, fs2, _ in tqdm(corrupted_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1, _ = clsmodel(auTmodel(fs1)[0])
            os2, _ = clsmodel(auTmodel(fs2)[0])

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            pl_loss = plloss(args, mem_label, idxes, os1) + plloss(args, mem_label, idxes, os2)

            loss = nucnm_loss + pl_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs1.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_pl_loss += pl_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Inferencing...')
        accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(corrupted_set)}')
        if accuracy >= max_accu:
            max_accu = accuracy
            # torch.save(auTmodel.state_dict(), os.path.join(args.output_path, f'{arch}-{constants.dataset_dic[args.dataset]}-auT-{constants.corruption_dic[args.corruption_type]}-{args.corruption_level}-pl.pt'))
            # torch.save(clsmodel.state_dict(), os.path.join(args.output_path, f'{arch}-{constants.dataset_dic[args.dataset]}-cls-{constants.corruption_dic[args.corruption_type]}-{args.corruption_level}-pl.pt'))

        wandb_run.log(
            data={
                'Loss/ttl_loss': ttl_loss / ttl_size,
                'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
                'Loss/PL loss': ttl_pl_loss / ttl_size,
                'Adaptation/accuracy': accuracy,
                'Adaptation/LR': learning_rate,
                'Adaptation/max_accu': max_accu,
            }, step=epoch, commit=True
        )