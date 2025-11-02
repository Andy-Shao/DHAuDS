import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Resize, Normalize, RandomCrop

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.acousticDataset import ReefSet
from lib.dataset import MultiTFDataset, batch_store_to, mlt_load_from, GpuMultiTFDataset
from lib.dataset import mlt_store_to
from lib.component import AudioClip, AudioPadding, OneHot2Index, time_shift
from lib.corruption import DynPSH
from CoNMix.lib.utils import Components, ExpandChannel, cal_norm, BiSet_Idx
from CoNMix.lib.plr import plr
from CoNMix.lib.loss import soft_CE, SoftCrossEntropyLoss, Entropy
from CoNMix.SpeechCommandsV2.train import load_models
from CoNMix.SpeechCommandsV2.STDA import load_origin_stat, build_optim, lr_scheduler, obtain_label
from CoNMix.ReefSet.train import inference

def rs_corrupt_data(args:argparse.Namespace) -> tuple[Dataset, Dataset]:
    from lib.corruption import corrupt_data

    n_mels=123
    hop_length=250
    if args.corruption_type == 'TST':
        test_set = corrupt_data(
            orgin_set=ReefSet(
                root_path=args.dataset_root_path, mode='test', include_rate=False, label_mode='single', label_tf=OneHot2Index()
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
        test_set = MultiTFDataset(
            dataset=test_set, tfs=[
                Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False)
                ])
            ]
        )
    else: 
        test_set = corrupt_data(
            orgin_set=ReefSet(
                root_path=args.dataset_root_path, mode='test', include_rate=False,
                data_tf=Components(transforms=[
                    AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
                    AudioClip(max_length=args.audio_length, mode='head', is_random=False)
                ]), label_mode='single', label_tf=OneHot2Index()
            ), corruption_level=args.corruption_level, corruption_type=args.corruption_type, enq_path=args.noise_path,
            sample_rate=args.sample_rate, end_path=args.noise_path, ensc_path=args.noise_path
        )
    
    # origin and weak set
    org_ds_path = os.path.join(args.cache_path, args.dataset, 'origin')
    index_file_name = 'metaInfo.csv'
    batch_store_to(
        data_loader=DataLoader(dataset=test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=8), 
        root_path=org_ds_path, index_file_name=index_file_name, f_num=1, is_one_hot_label=False
    )
    org_tf = [
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80.),
        ExpandChannel(out_channel=3),
        Resize((224, 224), antialias=False)
    ]
    if args.normalized:
        print('calculating original mean and standard deviation')
        org_set = mlt_load_from(
            root_path=org_ds_path, index_file_name=index_file_name, data_tfs=[Components(transforms=org_tf)]
        )
        org_mean, org_std = cal_norm(loader=DataLoader(dataset=org_set, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers))
        org_tf.append(Normalize(mean=org_mean, std=org_std))
    org_set = mlt_load_from(
        root_path=org_ds_path, index_file_name=index_file_name, data_tfs=[Components(transforms=org_tf)]
    )

    # strong augmentation set
    str_ds_path = os.path.join(args.cache_path, args.dataset, 'strong')
    str_set = GpuMultiTFDataset(
        dataset=mlt_load_from(
            root_path=org_ds_path, index_file_name=index_file_name
        ), 
        tfs=[Components(transforms=[
            # PitchShift(sample_rate=args.sample_rate, n_steps=4, n_fft=512),
            DynPSH(sample_rate=args.sample_rate, min_steps=4, max_steps=4, is_bidirection=False)
        ])], maintain_cpu=True
    )
    mlt_store_to(
        dataset=str_set, root_path=str_ds_path, index_file_name=index_file_name,
        data_tfs=[
            time_shift(shift_limit=.25, is_random=True, is_bidirection=True)
        ]
    )
    str_tf = [
        MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        AmplitudeToDB(top_db=80.),
        ExpandChannel(out_channel=3),
        Resize((256, 256), antialias=False),
        RandomCrop(224)
    ]
    if args.normalized:
        print('calculating strong augmentation mean and standard deviation')
        str_set = mlt_load_from(
            root_path=str_ds_path, index_file_name=index_file_name, 
            data_tfs=[Components(transforms=str_tf)]
        )
        str_mean, str_std = cal_norm(loader=DataLoader(dataset=str_set, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers))
        str_tf.append(Normalize(mean=str_mean, std=str_std))
    str_set = mlt_load_from(
        root_path=str_ds_path, index_file_name=index_file_name, 
        data_tfs=[Components(transforms=str_tf)]
    )

    return org_set, BiSet_Idx(set1=org_set, set2=str_set)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--noise_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    # CoNMix
    ap.add_argument('--modelF_weight_path', type=str)
    ap.add_argument('--modelB_weight_path', type=str)
    ap.add_argument('--modelC_weight_path', type=str)
    ap.add_argument('--max_epoch', type=int, default=100, help="max iterations")
    ap.add_argument('--interval', type=int, default=100)

    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--cls_par', type=float, default=0.2, help='lambda 2 | Pseudo-label loss capable')
    ap.add_argument('--cls_mode', type=str, default='soft_ce', choices=['logsoft_ce', 'soft_ce', 'logsoft_nll'])
    ap.add_argument('--alpha', type=float, default=0.9)
    ap.add_argument('--const_par', type=float, default=0.2, help='lambda 3')
    ap.add_argument('--ent_par', type=float, default=1.3)
    ap.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')
    ap.add_argument('--ent', action='store_true')
    ap.add_argument('--gent', action='store_true')

    ap.add_argument('--lr_decay1', type=float, default=0.1)
    ap.add_argument('--lr_decay2', type=float, default=1.0)
    ap.add_argument('--lr_gamma', type=int, default=30)
    ap.add_argument('--lr_momentum', type=float, default=.9)

    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    ap.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    ap.add_argument('--plr', type=int, default=1, help='Pseudo-label refinement')
    ap.add_argument('--sdlr', type=int, default=1, help='lr_scheduler capable')
    ap.add_argument('--initc_num', type=int, default=1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--backup_weight', type=int, default=0)

    ap.add_argument('--normalized', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'CoNMix'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'STDA')
    make_unless_exits(args.output_path)
    make_unless_exits(args.dataset_root_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    org_set, adpt_set = rs_corrupt_data(args=args)
    test_loader = DataLoader(
        dataset=org_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, 
        num_workers=args.num_workers
    )

    # build mode & load pre-train weight
    modelF, modelB, modelC = load_models(args)
    load_origin_stat(args, modelF=modelF, modelB=modelB, modelC=modelC)

    optimizer = build_optim(args, modelF=modelF, modelB=modelB, modelC=modelC)
    max_iter = args.max_epoch * len(adpt_loader)
    interval_iter = max_iter // args.interval
    iter = 0

    print('Beforing Adaptation')
    roc_auc = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=test_loader, device=args.device, class_num=args.class_num)
    print(f'Original test Mean ROC-AUC is: {roc_auc:.4f}')

    print('STDA Starting')
    max_roc_auc = 0.
    for epoch in range(args.max_epoch):
        if args.early_stop > 0 and epoch-1 == args.early_stop:
            break
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        ttl_loss = 0.
        ttl_cls_loss = 0.; ttl_const_loss = 0.; ttl_fbnm_loss = 0.; ttl_im_loss = 0.
        ttl_num = 0
        epoch_flag = True

        print('Adapting...')
        modelF.train(); modelB.train(); modelC.train()
        for weak_features, strong_features, _, idxes in tqdm(adpt_loader):
            batch_size = weak_features.shape[0]
            if epoch_flag and args.cls_par >= 0:
                epoch_flag = False
                modelF.eval(); modelB.eval(); modelC.eval()
                # print('Starting to find Pseudo Labels! May take a while :)')
                # test loader same as target but has 3*batch_size compared to target and train
                mem_label, soft_output, dd, mean_all_output, actual_label = obtain_label(loader=test_loader, modelF=modelF, modelB=modelB, modelC=modelC, args=args, step=epoch)

                if args.plr:
                    if iter == 0:
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
                modelF.train(); modelB.train(); modelC.train()
            iter += 1

            features = torch.cat([weak_features, strong_features], dim=0).to(args.device)

            outputs_B = modelB(modelF(features))
            outputs = modelC(outputs_B)

            # Pseudo-label cross-entropy loss
            if args.cls_par > 0:
                with torch.no_grad():
                    pred = mem_label[idxes]
                if args.cls_mode == 'logsoft_ce':
                    classifier_loss = SoftCrossEntropyLoss(outputs[0:batch_size], pred)
                    classifier_loss = torch.mean(classifier_loss)
                elif args.cls_mode == 'soft_ce':
                    softmax_output = nn.Softmax(dim=1)(outputs[0:batch_size])
                    classifier_loss = nn.CrossEntropyLoss()(softmax_output, pred)
                elif args.cls_mode == 'logsoft_nll':
                    softmax_output = nn.LogSoftmax(dim=1)(outputs[0:batch_size])
                    _, pred = torch.max(pred, dim=1)
                    classifier_loss = nn.NLLLoss(reduction='mean')(softmax_output, pred)
                classifier_loss = args.cls_par * classifier_loss
            else:
                classifier_loss = torch.tensor(.0).cuda()
            
            # fbnm -> Nuclear-norm Maximization loss
            if args.fbnm_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_output,2),dim=0)), descending=True)
                fbnm_loss = - torch.mean(list_svd[:min(softmax_output.shape[0],softmax_output.shape[1])])
                fbnm_loss = args.fbnm_par*fbnm_loss
            else:
                fbnm_loss = torch.tensor(.0).cuda()

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs)		# find number of psuedo sample per class for handling class imbalance for entropy maximization
                entropy_loss = torch.mean(Entropy(softmax_out))#softmax_outputs_stg = nn.Softmax(dim=1)(outputs_stg)
                en_loss = entropy_loss.item()
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    gen_loss = gentropy_loss.item()
                    entropy_loss -= gentropy_loss
                im_loss = entropy_loss * args.ent_par
            else:
                im_loss = torch.tensor(0.0).cuda()

            # Consist loss -> soft cross-entropy loss
            if args.const_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                expectation_ratio = mean_all_output/torch.mean(softmax_output[0:batch_size],dim=0)
                with torch.no_grad():
                    soft_label_norm = torch.norm(softmax_output[0:batch_size]*expectation_ratio,dim=1,keepdim=True) #Frobenius norm
                    soft_label = (softmax_output[0:batch_size]*expectation_ratio)/soft_label_norm
                consistency_loss = args.const_par*torch.mean(soft_CE(softmax_output[batch_size:],soft_label))
                cs_loss = consistency_loss.item()
            else:
                consistency_loss = torch.tensor(.0).cuda()
            
            total_loss = classifier_loss + fbnm_loss + im_loss + consistency_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            ttl_loss += total_loss.item()
            ttl_cls_loss += classifier_loss.item()
            ttl_const_loss += consistency_loss.item()
            ttl_fbnm_loss += fbnm_loss.item()
            ttl_im_loss += im_loss.item()
            ttl_num += weak_features.shape[0]

            if iter % interval_iter == 0:
                if args.sdlr:
                    lr_scheduler(optimizer, iter_num=iter, max_iter=max_iter, gamma=args.lr_gamma, step=epoch, momentum=args.lr_momentum)

        print('Inferecing...')
        roc_auc = inference(modelF=modelF, modelB=modelB, modelC=modelC, data_loader=test_loader, device=args.device, class_num=args.class_num)
        if roc_auc > max_roc_auc:
            max_roc_auc = roc_auc
            if args.backup_weight == 0:
                torch.save(modelF.state_dict(), os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}-modelF.pt'))
                torch.save(modelB.state_dict(), os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}-modelB.pt'))
                torch.save(modelC.state_dict(), os.path.join(args.output_path, f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}-modelC.pt'))
        wandb.log({'Adaptation/ROC-AUC': roc_auc, 'Adaptation/Max_ROC-AUC': max_roc_auc}, step=epoch)
        print(f'Adaptation/ROC-AUC: {roc_auc:.4f}, Adaptation/Max_ROC-AUC: {max_roc_auc:.4f}, Sample size: {len(org_set)}')
        wandb.log({
            "Loss/ttl_loss":ttl_loss / ttl_num, 
            "Loss/PL loss":ttl_cls_loss / ttl_num, 
            "Loss/Consistency loss":ttl_const_loss / ttl_num, 
            "Loss/Nuclear-norm loss":ttl_fbnm_loss / ttl_num,
            "Loss/IM loss":ttl_im_loss / ttl_num,
        }, step=epoch, commit=True)