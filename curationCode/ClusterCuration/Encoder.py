"""https://music-classification.github.io/tutorial/part3_supervised/tutorial.html"""
import librosa.feature
from torch import nn
import torchaudio
from torch import Tensor
import torch
from torch.nn import functional as F
from typing import Callable, Optional
from torchaudio.models import Conformer


"""https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py"""

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """This function creates a 3x3 convolution with padding.
    Input:
        - in_planes: the number of channels in the input
        - out_planes: the number of channels in the output
        - stride: the stride of the convolution
        - groups: the number of blocked connections from the input channel
        - dilation: the spacing between the kernel elements
    Output:
        2 Dimensional Convolutional layer"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 464,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Conv_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, shape, padding=1)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # out = self.dropout(out)
        return out


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2, dropout=0.1):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size= 3, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm2d(3)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        # out = self.maxpool(out)
        # out = self.dropout(out)
        # out = self.batchnorm(out)
        return out

class MultiLayerPerceptron(nn.Module):
    # def __init__(self, in_features, out_features):
    #     super(MultiLayerPerceptron, self).__init__()
    #     self.linear = nn.Linear(in_features, in_features)
    #     self.relu = nn.ReLU()
    #     self.linear2 = nn.Linear(in_features, out_features)
    #     self.dropout = nn.Dropout(0.2)
    def __init__(self, in_features, out_features):
        super(MultiLayerPerceptron, self).__init__()
        self.linear = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.input_bn1 = nn.BatchNorm1d(in_features)
        self.input_bn2 = nn.BatchNorm1d(out_features)

    # def forward(self, x):
    #     out = self.linear(x)
    #     out = self.relu(out)
    #     out = self.linear2(out)
    def forward(self, x):
        out = self.linear(x)
        out = self.input_bn1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.input_bn2(out)
        return out

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHead, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x):
        out = self.linear(x)
        return out

class LinearHead_Added(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHead, self).__init__()
        self.input_bn = nn.BatchNorm2d(in_features)
        self.linear = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x):
        out = self.input_bn(x)
        out = self.linear(out)
        return out

class Expander(nn.Module):
    def __init__(self, input_dim):
       super(Expander, self).__init__()
       self.layer1 = nn.Linear(input_dim, 8192)
       self.bn = nn.BatchNorm1d(8192)
       self.relu = nn.ReLU(True)
       self.layer2 = nn.Linear(8192, 8192, bias=False)

    def forward(self,x):
       out = self.layer1(x)
       out = self.bn(out)
       out = self.relu(out)
       out = self.layer2(out)
       return out


class Preprocess_MelSpec(object):

    def __init__(self, sample_rate=16000, n_fft=1024,f_min=0.0,f_max=8000,num_mels=128):
        self.samplerate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.num_mels = num_mels

    def _convert_to_db(self, melspec):
        db_melspec = librosa.amplitude_to_db(melspec)
        return db_melspec

    def __call__(self, audio):
        melspec = librosa.feature.melspectrogram(y=audio.numpy(), sr=self.samplerate, n_fft=self.n_fft, n_mels = self.num_mels,
                                                 fmin = self.f_min, fmax=self.f_max)
        melspec = self._convert_to_db(melspec)
        return torch.from_numpy(melspec)


# class Encoder(nn.Module):
#     def __init__(self, num_channels=3,
#                        sample_rate=16000,
#                        n_fft=1024,
#                        f_min=0.0,
#                        f_max=8000,
#                        num_mels=128,
#                        num_classes=2,
#                         Baseline=False,
#                         transformed=False,
#                         num_heads=4,
#                         ffn_dim=128,
#                         num_layers=4,
#                         depthwise_conv_kernel_size=31):
#         super(Encoder, self).__init__()
#         self.MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
#                                                n_mfcc=13,
#                                                melkwargs={"n_fft": 1024, "n_mels": 128, "f_min": f_min, "f_max":f_max})
#         self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
#                                                             n_fft=n_fft,
#                                                             f_min=f_min,
#                                                             f_max=f_max,
#                                                             n_mels=num_mels)
#         self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
#         self.num_classes = num_classes
#         self.Baseline = Baseline
#         self.transformed = transformed
#         self.input_bn = nn.BatchNorm2d(1)
#
#         # Move layer initialization here to avoid loading them at import time
#         self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))
#
#     def _initialize_layers(self):
#         """Lazy layer initialization"""
#         self.layer2 = Conformer(input_dim=self.num_mels, num_heads=4, ffn_dim=128, num_layers=4,
#                                 depthwise_conv_kernel_size=31)
#         self.layer3 = Expander(2048)
#         self.layer4 = LinearHead(128, 4)
#         self.fc = nn.Linear(8064, 2048)
#
#     def forward(self, wav):
#         if not hasattr(self, 'layer2'):  # Check if layers have been initialized
#             self._initialize_layers()  # Initialize the layers when forward pass happens
#
#         out = self.melspec(wav)
#         out = self.amplitude_to_db(out)
#         out = self.input_bn(out)
#         out, lengths = self.layer2(out.squeeze(1).transpose(1, 2), torch.tensor([313]).repeat(out.shape[0]))
#         out = torch.flatten(out, 1)
#         out_repr = self.fc(out)
#         out_contrast = self.layer3(out_repr)
#
#         return out_contrast, out_repr


class Encoder(nn.Module):
    def __init__(self, num_channels=3,
                       sample_rate=16000,
                       n_fft=1024,
                       f_min=0.0,
                       f_max=8000,
                       num_mels=128,
                       num_classes=2,
                        Baseline=False,
                        transformed=False,
                        num_heads=4,
                        ffn_dim=128,
                        num_layers=4,
                        depthwise_conv_kernel_size=31):
        super(Encoder, self).__init__()
        #
        # window = torch.hann_window(window_length=n_fft)

        # mel spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=num_mels)
        # # self.melspec = Preprocess_MelSpec(sample_rate=sample_rate,
        # #                                                     n_fft=n_fft,
        # #                                                     f_min=f_min,
        # #                                                     f_max=f_max,
        # #                                                     num_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # self.melspec = MelSpectrogram(sr=sample_rate, n_fft=1024, fmin=f_min, fmax=f_max, trainable_STFT=False, trainable_mel =False)
        self.MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                               n_mfcc=13,
                                               melkwargs={"n_fft": 1024, "n_mels": 128, "f_min": f_min, "f_max":f_max})
        # self.norm = Normalize(6.069, 49.7597, True)
        self.num_classes = num_classes
        self.Baseline = Baseline
        self.input_bn = nn.BatchNorm2d(1)
        self.transformed = transformed
        # self.input_bn = nn.BatchNorm1d(1)

        # convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))

        # self.layer2 = ResNet(layers=[2, 2, 2, 2], num_classes=2048, block=BasicBlock)
        self.layer2 = Conformer(input_dim=num_mels, num_heads=4, ffn_dim=128, num_layers=4,depthwise_conv_kernel_size=31)
        # self.layer2 = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        # self.layer2.classifier = Identity()
        # self.layer2 = UNet(1)

        # self.layer3 = MultiLayerPerceptron(1536, 128)
        # self.layer3 = MultiLayerPerceptron(2048, 128)
        # self.layer3 = MultiLayerPerceptron(2048, num_classes)
        # self.layer3 = MultiLayerPerceptron(768, 128)
        self.layer3 = Expander(2048)

        # if self.Baseline:
        # self.layer4 = LinearHead(2048, 4)
        # self.layer4 = LinearHead(768, num_classes)
        self.layer4 = LinearHead(128, 4)
        # self.layer4 = nn.Linear(128, 12)

        self.fc = nn.Linear(8064, 2048)
        # self.fc = nn.Linear(20096, 2048)
        # self.fc = nn.Linear(60032, 2048)
        # self.fc = LinearHead(768, 2*sample_rate)



    def forward(self, wav):
        print(wav.shape)
        # input Preprocessing
        out = self.melspec(wav)
        out = self.amplitude_to_db(out)
        # out = self.feature_extractor(wav)

        out = self.input_bn(out)

        # convolutional layers
        # out = out.repeat(1,3,1,1)
        # out = self.layer1(out)
        # out_repr = self.layer2(out)
        # out_contrast = self.layer3(out_repr)

        # Conformer layer
        # out, lengths = self.layer2(out.squeeze(1).transpose(1,2), torch.tensor([63]).repeat(out.shape[0]))
        out, lengths = self.layer2(out.squeeze(1).transpose(1, 2), torch.tensor([313]).repeat(out.shape[0]))
        # try:
        #    out, lengths = self.layer2(out.squeeze(1).transpose(1,2), torch.tensor([63]).repeat(out.shape[0]))
        # except:
        #    out, lengths = self.layer2(out.squeeze(1).transpose(1,2), torch.tensor([32]).repeat(out.shape[0]))
        out = torch.flatten(out, 1)
        # out_repr = self.fc(out)
        # fc_layer = nn.Linear(out.shape[1], 2048).to('cuda:0')
        fc_layer = nn.Linear(out.shape[1], 2048)
        out_repr = fc_layer(out)
        out_contrast = self.layer3(out_repr)
        # out_contrast = self.layer4(out_repr)
        # model = nn.Sequential(self.melspec,
        #                       self.amplitude_to_db,
        #                       self.input_bn,
        #                       self.layer1,
        #                       self.layer2,
        #                       self.layer3)
        # if self.Baseline:
        #     out_sup = self.layer4(out)

        # AST layers
        # out = self.layer2(wav)
        #
        # out_contrast = self.layer3(out.logits)
        #
        # out_repr = self.fc(out.logits)
        #
        # UNet
        # out_repr, out = self.layer2(out)
        # flatten = nn.Flatten()
        # out = flatten(out)
        # print(out.shape)
        # out_contrast = self.layer3(out)

        return out_contrast, out_repr
        # return model(wav)

class NTXentLossFunction():
    def __init__(self, temperature):
        super(NTXentLossFunction, self).__init__()
        self.temperature = temperature



    def _create_embedding(self, samples, pos_pairs):
        embeddings = torch.cat([samples, pos_pairs], dim=0)
        # print('embeddings: ', embeddings.shape)
        labels_1 = torch.arange(samples.shape[0])

        labels_2 = torch.add(labels_1, len(samples))
        labels = torch.cat([labels_2, labels_1], dim=0)
        # print(labels.shape)
        # print(labels)
        return embeddings, labels
    def nt_xent_loss(self, x, target, temperature):
        assert len(x.size()) == 2

        # Cosine similarity
        xcs = F.cosine_similarity(x[None, :, :], x[:, None, :], dim=-1)
        xcs[torch.eye(x.size(0)).bool()] = float("-inf")
        xcs = xcs.to(device='cpu')
        target = target.to(device='cpu')
        # Standard cross entropy loss
        return F.cross_entropy(xcs / temperature, target, reduction="mean")

    def __call__(self, samples, pos_pairs):
        embeddings, labels = self._create_embedding(samples, pos_pairs)
        embeddings = embeddings.to(device='cpu')
        labels = labels.to(device='cpu')
        loss = self.nt_xent_loss(embeddings, labels, self.temperature)
        return loss

class VICRegLossFunction():
    def __init__(self, batch_size, sim_coeff, std_coeff, cov_coeff, num_features):
        super(VICRegLossFunction, self).__init__()
        self.batch_size = batch_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.num_features = num_features
        # self.expander = Expander()

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        # print(x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten().shape)
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _on_diagonal(self, x):
        return torch.diagonal(x, 0)

    def __call__(self, x, y):
        # repr_loss = F.mse_loss(x, y)
        repr_loss = -F.cosine_similarity(x, y).mean()
        # theta = F.cosine_similarity(x, y)
        # repr_loss = -(torch.linalg.norm(x, dim=1).pow(0.5) * torch.linalg.norm(y, dim=1).pow(0.5) * theta).mean()
        # # print(repr_loss.shape)
        # repr_loss = -(x * y).sum (dim = 1).mean()

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)



        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)


        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # cov_x = (x.T @ x) / (self.batch_size - 1)
        # cov_y = (y.T @ y) / (self.batch_size - 1)
        # cov_loss = self._off_diagonal(cov_x).pow_(2).sum().div(
        #     self.num_features
        # ) + self._off_diagonal(cov_y).pow_(2).sum().div(self.num_features)


        # cov_m = (x.T @ y) / (self.batch_size - 1)

        # cov_loss = self._off_diagonal(cov_m).pow_(2).sum().div(
        #     self.num_features
        # )
        # cov_loss = self._off_diagonal(cov_m).pow_(2).sum().div(
        #     self.num_features
        # )
        cov_m = (x.T @ y) / (self.batch_size - 1)

        # cov_d = self._on_diagonal(cov_m).pow_(2).sum().div(
        #     self.num_features
        # )

        cov_loss = self._off_diagonal(cov_m).pow_(2).sum().div(self.num_features*(self.num_features-1))



        loss = (
                self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
        )
        # print('loss components: ', repr_loss, std_loss, cov_loss, loss)
        return loss, repr_loss, std_loss, cov_loss

class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        n_stft = int((1024 // 2) + 1)
        self.invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=16000, n_stft=n_stft)
        self.invers_transform.cuda()
        self.grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=1024)
        self.grifflim_transform.cuda()

    def forward(self, input, target):
        input = self.invers_transform(input)
        input = self.grifflim_transform(input).squeeze(1)
        # target = self.invers_transform(target)
        # total_loss = []
        # print(inputs.shape, targets.shape)
        # for input, target in zip(inputs, targets):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
            loss /= (12 - 6)
                # total_loss.append(loss.item())
                # print(total_loss)
        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = torch.div((anchor_dot_contrast - logits_min), (logits_max - logits_min))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Weighted_NTXentLossFunction(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(Weighted_NTXentLossFunction, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def _create_weights(self, samples):
        # Weights based on eucledian distance
        # print('weights')
        # print(torch.cdist(samples, samples, p=2).div(samples.shape[0]))
        return torch.cdist(samples, samples, p=2)


    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = torch.div((anchor_dot_contrast - logits_min), (logits_max - logits_min))

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        weights = self._create_weights(anchor_feature)
        exp_logits = weights * torch.exp(logits) * logits_mask
        # exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CrossCorrelationLossFunction():
    def __init__(self, lambd):
        super(CrossCorrelationLossFunction, self).__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(128, affine=False)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def __call__(self, samples, pos_pairs):
        c = self.bn(samples).T @ self.bn(pos_pairs)

        # sum the cross-correlation matrix between all gpus
        c.div_(samples.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
