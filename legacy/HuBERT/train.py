import argparse
from tqdm import tqdm
import random
import os

import torchaudio
from torchvision.transforms import Compose
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from lib.model import Classifier
from lib.spdataset import SpeechCommandsV2

class ReduceChannel(nn.Module):
    def __init__(self):
        super(ReduceChannel, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.squeeze(x, dim=0)

class AudioPadding(nn.Module):
    def __init__(self, max_length:int, sample_rate:int, random_shift:bool=False):
        super(AudioPadding, self).__init__()
        self.max_length = max_length
        self.random_shift = random_shift

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        l = self.max_length - x.shape[1]
        if l > 0:
            if self.random_shift:
                head = random.randint(0, l)
                tail = l - head
            else:
                head = l // 2
                tail = l - head
            x = pad(x, (head, tail), mode='constant', value=0.)
        return x

def build_model(args:argparse.Namespace) -> tuple[torchaudio.models.Wav2Vec2Model, Classifier]:
    bundle = torchaudio.pipelines.HUBERT_BASE
    hubert = bundle.get_model().to(device=args.device)
    classifier = Classifier(class_num=args.class_num, embed_size=768).to(device=args.device)
    return hubert, classifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--max_epoch', type=int, default=30)
    ap.add_argument('--output_path', type=str, default='.')
    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.class_num = 35
    args.sample_rate = 16000
    ############################################################

    train_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='train', include_rate=False,
        data_tf=Compose(transforms=[
            AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=True),
            ReduceChannel()
        ])
    )
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    hubert, clsModel = build_model(args=args)
    optimizer = optim.SGD(lr=args.lr, params=clsModel.parameters())
    loss_fn = nn.CrossEntropyLoss().to(device=args.device)

    hubert.eval()
    clsModel.train()
    ttl_corr = 0.
    ttl_size = 0.
    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            hiddent_fs = hubert.extract_features(features)[0]
            hiddent_fs = hiddent_fs[-1]
            outputs = clsModel(hiddent_fs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            ttl_size += labels.shape[0]
            _, preds = torch.max(input=outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
        ttl_acc = ttl_corr/ttl_size * 100.
        print(f'Accuracy is: {ttl_acc:.4f}%, sample size is: {len(train_set)}')

        if max_accu <= ttl_acc:
            max_accu == ttl_acc
            torch.save(obj=clsModel.state_dict(), f=os.path.join(args.output_path, 'clsModel.pt'))