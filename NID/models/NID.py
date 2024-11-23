from torch import nn
import torch
import collections
import numpy as np

from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils
from ..models import conv_stft

def UnwrapPhase(input):
    input = input.to('cpu').numpy()
    input = np.unwrap(input, axis=-2)
    output = input
    output = torch.from_numpy(output)
    output = output.to('cuda')
    return output

class SelfConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size,
               stride=1, use_bias=True, padding='same', activation=None):
    super(SelfConv2d, self).__init__()
    assert padding.lower() in ['same', 'valid'], 'padding must be same or valid.'
    if padding.lower() == 'same':
      if type(kernel_size) is int:
        padding_nn = kernel_size // 2
      else:
        padding_nn = []
        for kernel_s in kernel_size:
          padding_nn.append(kernel_s // 2)
    self.conv2d_fn = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=use_bias, padding=padding_nn)
    self.act = None if activation is None else activation()

  def forward(self, feature_in):
    out = feature_in
    out = self.conv2d_fn(out)
    if self.act is not None:
      out = self.act(out)
    return out
  

class DTFA(nn.Module):
  def __init__(self, in_channels, out_channels, frequency_dim):
    super(DTFA, self).__init__()
    self.cov1d_d_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=1)
    self.cov1d_d_2 = nn.Conv1d(out_channels, in_channels, kernel_size=1, dilation=2)
    self.conv2d_fn = nn.Conv2d(in_channels, 1, [1,1], stride=1, bias=True, padding="same")
    self.conv2d_fn_2 = nn.Conv2d(23, in_channels, [1,1], stride=1, bias=True, padding="same")
    self.relu = nn.ReLU()

  def forward(self, feature_in):
    out_zt = torch.mean(feature_in, dim=2, keepdim=True)
    out_zt = torch.squeeze(out_zt, dim=2)
    out_zt = self.cov1d_d_1(out_zt)
    out_zt = self.relu(out_zt)
    out_zt = self.cov1d_d_2(out_zt)
    out_zt = torch.sigmoid(out_zt)
    out_zt = torch.unsqueeze(out_zt, 2)
    out_zf = torch.mean(feature_in, dim=3, keepdim=True)
    out_zf = torch.squeeze(out_zf, dim=3)
    out_zf = self.cov1d_d_1(out_zf)
    out_zf = self.relu(out_zf)
    out_zf = self.cov1d_d_2(out_zf)
    out_zf = torch.sigmoid(out_zf)
    out_zf = torch.unsqueeze(out_zf, 3)
    out = torch.matmul(out_zf, out_zt)
    out = self.conv2d_fn(out)
    dnum = 300
    gap = 1 / dnum
    cnt = 0
    for i in range(1, dnum):
      cnt = cnt + i * gap
      if cnt < 1:
        c = torch.tensor([cnt]).to(out.device)
        dcomp_tmp = torch.where(out > c, torch.tensor(1).to(out.device), torch.tensor(0).to(out.device))
        dcomp_tmp = torch.mul(dcomp_tmp, out)
        if i == 1:
            dcomp = dcomp_tmp
        else:
            dcomp = torch.cat((dcomp, dcomp_tmp), dim=1)
      else:
        break
    out = self.conv2d_fn_2(dcomp)
    out = torch.mul(out, feature_in)
    return out

class BatchNormAndActivate(nn.Module):
  def __init__(self, channel, activation=nn.ReLU):
    super(BatchNormAndActivate, self).__init__()
    self.bn_layer = nn.BatchNorm2d(channel)
    self.activate_fn = None if activation is None else activation()

  def forward(self, fea_in):
    out = self.bn_layer(fea_in)
    if self.activate_fn is not None:
      out = self.activate_fn(out)
    return out

  def get_bn_weight(self):
    return self.bn_layer.parameters()


class Encoder_M(nn.Module):
  def __init__(self, in_channels, out_channels, conv2d_activation=None):
    super(Encoder_M, self).__init__()

    self.conv_1 = nn.Conv2d(in_channels, 4, [4, 4], dilation=1, padding="same")
    self.bn_fn_1 = BatchNormAndActivate(4, activation=conv2d_activation)

    self.conv_2 = nn.Conv2d(6, 8, [3, 3], dilation=2, padding="same")
    self.bn_fn_2 = BatchNormAndActivate(8, activation=conv2d_activation)

    self.conv_3 = nn.Conv2d(16, 16, [2, 2], dilation=3, padding="same")
    self.bn_fn_3 = BatchNormAndActivate(16, activation=conv2d_activation)

    self.conv_4 = nn.Conv2d(40, 32, [1, 1], dilation=4, padding="same")
    self.bn_fn_4 = BatchNormAndActivate(32, activation=conv2d_activation)


  def forward(self, feature_in):
    out1 = self.conv_1(feature_in)
    out1 = self.bn_fn_1(out1)
    out1 = torch.cat((out1, feature_in), dim=1)

    out2 = self.conv_2(out1)
    out2 = self.bn_fn_2(out2)
    out2 = torch.cat((out2, out1, feature_in), dim=1)

    out3 = self.conv_3(out2)
    out3 = self.bn_fn_3(out3)
    out3 = torch.cat((out3, out2, out1, feature_in), dim=1)

    out4 = self.conv_4(out3)
    out4 = self.bn_fn_4(out4)
    out4 = torch.cat((out4, out3, out2, out1, feature_in), dim=1)

    return out4
  

class Encoder_P(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Encoder_P, self).__init__()
    self.conv_1 = nn.Conv2d(in_channels, 2, [4, 4], dilation=1, padding="same")
    self.conv_2 = nn.Conv2d(3, 4, [3, 3], dilation=2, padding="same")
    self.conv_3 = nn.Conv2d(8, 8, [2, 2], dilation=3, padding="same")
    self.conv_4 = nn.Conv2d(20, 16, [1, 1], dilation=4, padding="same")


  def forward(self, feature_in):
    out = feature_in
    out = UnwrapPhase(out)
    prev_element = out[:, :, :-1, :]
    next_element = out[:, :, 1:, :]
    diff_tensor = next_element - prev_element
    padded_tensor = torch.nn.functional.pad(diff_tensor, pad=(0, 0, 1, 0), mode='constant', value=0) # 填0
    squared_tensor = torch.pow(padded_tensor, 2)
    out1 = self.conv_1(squared_tensor)
    out1 = torch.cat((out1, squared_tensor), dim=1)
    out2 = self.conv_2(out1)
    out2 = torch.cat((out2, out1, squared_tensor), dim=1)
    out3 = self.conv_3(out2)
    out3 = torch.cat((out3, out2, out1, squared_tensor), dim=1)
    out4 = self.conv_4(out3)
    out4 = torch.cat((out4, out3, out2, out1, squared_tensor), dim=1)
    return out4

class DTFABlock(nn.Module):
  def __init__(self, frequency_dim, channel_in_out, channel_attention=5):
    super(DTFABlock, self).__init__()
    self.frequency_dim = frequency_dim
    self.channel_out = channel_in_out
    self.tfa = DTFA(channel_in_out, channel_attention, frequency_dim)
    self.out_conv2d = SelfConv2d(channel_in_out*2, channel_in_out, [1, 1], padding="same")
    self.out_conv2d_bna = BatchNormAndActivate(channel_in_out)

  def forward(self, feature_in):
    att_out = self.tfa(feature_in)
    concated_out = torch.cat([feature_in, att_out], 1)
    out = self.out_conv2d(concated_out)
    out = self.out_conv2d_bna(out)
    return out


class InfoCommunicate(nn.Module):
  def __init__(self, channel_in, channel_out, activate_fn=nn.Tanh):
    super(InfoCommunicate, self).__init__()
    self.conv2d = SelfConv2d(channel_in, channel_out, [1, 1], padding="same")
    self.activate_fn = None if activate_fn is None else activate_fn()

  def forward(self, feature_x1, feature_x2):
    out = self.conv2d(feature_x2)
    if self.activate_fn is not None:
      out = self.activate_fn(out)
    out_multiply = torch.mul(feature_x1, out)
    return out_multiply


class TwoStreamBlock(nn.Module):
  def __init__(self, frequency_dim, channel_in_out_A, channel_in_out_P):
    super(TwoStreamBlock, self).__init__()
    self.sA1_pre_FTB = DTFABlock(frequency_dim, channel_in_out_A)
    self.sA2_conv2d = SelfConv2d(channel_in_out_A, channel_in_out_A, [5, 5], padding="same")
    self.sA2_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
    self.sA3_conv2d = SelfConv2d(channel_in_out_A, channel_in_out_A, [1, 25], padding="same")
    self.sA3_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
    self.sA4_conv2d = SelfConv2d(channel_in_out_A, channel_in_out_A, [5, 5], padding="same")
    self.sA4_conv2d_bna = BatchNormAndActivate(channel_in_out_A)
    self.sA5_post_FTB = DTFABlock(frequency_dim, channel_in_out_A)
    self.sA6_info_communicate = InfoCommunicate(channel_in_out_P, channel_in_out_A)
    self.sP1_conv2d_before_LN = nn.LayerNorm([frequency_dim, channel_in_out_P])
    self.sP1_conv2d = SelfConv2d(channel_in_out_P, channel_in_out_P, [3, 5], padding="same")
    self.sP2_conv2d_before_LN = nn.LayerNorm([frequency_dim, channel_in_out_P])
    self.sP2_conv2d = SelfConv2d(channel_in_out_P, channel_in_out_P, [1, 25], padding="same")
    self.sP3_info_communicate = InfoCommunicate(channel_in_out_A, channel_in_out_P)

  def forward(self, feature_sA, feature_sP):
    sA_out = feature_sA
    sA_out = self.sA1_pre_FTB(sA_out)
    sA_out = self.sA2_conv2d(sA_out)
    sA_out = self.sA2_conv2d_bna(sA_out)
    sA_out = self.sA3_conv2d(sA_out)
    sA_out = self.sA3_conv2d_bna(sA_out)
    sA_out = self.sA4_conv2d(sA_out)
    sA_out = self.sA4_conv2d_bna(sA_out)
    sA_out = self.sA5_post_FTB(sA_out)
    sP_out = feature_sP
    sP_out = torch.transpose(sP_out, 1, 3)
    sP_out = self.sP1_conv2d_before_LN(sP_out)
    sP_out = torch.transpose(sP_out, 1, 3)
    sP_out = self.sP1_conv2d(sP_out)
    sP_out = torch.transpose(sP_out, 1, 3)
    sP_out = self.sP2_conv2d_before_LN(sP_out)
    sP_out = torch.transpose(sP_out, 1, 3)
    sP_out = self.sP2_conv2d(sP_out)
    sA_fin_out = self.sA6_info_communicate(sA_out, sP_out)
    sP_fin_out = self.sP3_info_communicate(sP_out, sA_out)
    sA_fin_out = sA_fin_out + feature_sA
    sP_fin_out = sP_fin_out + feature_sP

    return sA_fin_out, sP_fin_out


class Decoder_M(nn.Module):
  def __init__(self, frequency_dim, channel_sA):
    super(Decoder_M, self).__init__()
    self.p1_conv2d = nn.Conv2d(channel_sA, 48, [1, 1], dilation=1, padding="same")
    self.bn_fn_1 = BatchNormAndActivate(48, activation=nn.Sigmoid)
    self.p2_conv2d = nn.Conv2d(48, 24, [1, 1], dilation=2, padding="same")
    self.bn_fn_2 = BatchNormAndActivate(24, activation=nn.Sigmoid)
    self.p3_conv2d = nn.Conv2d(24, 12, [1, 1], dilation=3, padding="same")
    self.bn_fn_3 = BatchNormAndActivate(12, activation=nn.Sigmoid)
    self.p4_conv2d = nn.Conv2d(12, 6, [1, 1], dilation=4, padding="same")
    self.bn_fn_4 = BatchNormAndActivate(6, activation=nn.Sigmoid)
    self.p5_conv2d = nn.Conv2d(6, 3, [1, 1], dilation=5, padding="same")
    self.bn_fn_5 = BatchNormAndActivate(3, activation=nn.Sigmoid)
    self.p6_conv2d = nn.Conv2d(3, 1, [1, 1], dilation=6, padding="same")
    self.bn_fn_6 = BatchNormAndActivate(1, activation=nn.Sigmoid)

  def forward(self, feature_sA):
    out = feature_sA
    out = self.p1_conv2d(out)
    out = self.bn_fn_1(out)
    out = self.p2_conv2d(out)
    out = self.bn_fn_2(out)
    out = self.p3_conv2d(out)
    out = self.bn_fn_3(out)
    out = self.p4_conv2d(out)
    out = self.bn_fn_4(out)
    out = self.p5_conv2d(out)
    out = self.bn_fn_5(out)
    out = self.p6_conv2d(out)
    out = self.bn_fn_6(out)
    out = out.squeeze(1)
    return out


class Decoder_P(nn.Module):
  def __init__(self, channel_sP):
    super(Decoder_P, self).__init__()
    self.conv2d = nn.Conv2d(channel_sP, 24, [1, 1], dilation=1, padding="same")
    self.conv2d_2 = nn.Conv2d(24, 12, [1, 1], dilation=2, padding="same")
    self.conv2d_3 = nn.Conv2d(12, 6, [1, 1], dilation=3, padding="same")
    self.conv2d_4 = nn.Conv2d(6, 3, [1, 1], dilation=4, padding="same")
    self.conv2d_5 = nn.Conv2d(3, 2, [1, 1], dilation=5, padding="same")

  def forward(self, feature_sP:torch.Tensor):
    out = feature_sP
    out = self.conv2d(out)
    out = self.conv2d_2(out)
    out = self.conv2d_3(out)
    out = self.conv2d_4(out)
    out = self.conv2d_5(out)
    out_real = out[:, :1, :, :]
    out_imag = out[:, 1:, :, :]
    out_angle = torch.atan2(out_imag, out_real)
    normed_stft = torch.div(out, torch.sqrt(out_real**2+out_imag**2)+PARAM.stft_div_norm_eps)
    return normed_stft, out_angle.squeeze(1)


class WavFeatures(
    collections.namedtuple("WavFeatures",
                           ("wav_batch",
                            "stft_batch",
                            "mag_batch",
                            "angle_batch",
                            "normed_stft_batch",
                            ))):
  pass


class NET_NID_OUT(
    collections.namedtuple("NET_PHASEN_OUT",
                           ("mag_mask", "normalized_complex_phase", "angle"))):
  pass


class NetNID(nn.Module):
  def __init__(self):
    super(NetNID, self).__init__()
    sA_in_channel = {
      "stft":2,
      "mag":1,
    }[PARAM.stream_A_feature_type]
    sP_in_channel = {
      "stft":2,
      "normed_stft":2,
      "angle":1,
    }[PARAM.stream_P_feature_type]
    self.streamA_prenet = Encoder_M(sA_in_channel, PARAM.channel_A, conv2d_activation=nn.ReLU)
    self.streamP_prenet = Encoder_P(sP_in_channel, PARAM.channel_P)
    
    self.layers_TSB = nn.ModuleList()
    for i in range(1, PARAM.n_TSB+1):
      tsb_t = TwoStreamBlock(PARAM.frequency_dim, PARAM.channel_A, PARAM.channel_P)
      self.layers_TSB.append(tsb_t)
    
    self.streamA_postnet = Decoder_M(
        PARAM.frequency_dim, PARAM.channel_A)
    self.streamP_postnet = Decoder_P(PARAM.channel_P)

  def forward(self, mixed_wav_features:WavFeatures):
    sA_inputs = {
      "stft":mixed_wav_features.stft_batch,
      "mag":mixed_wav_features.mag_batch.unsqueeze(1),
    }[PARAM.stream_A_feature_type]
    sP_inputs = {
      "stft":mixed_wav_features.stft_batch,
      "normed_stft":mixed_wav_features.normed_stft_batch,
      "angle":mixed_wav_features.angle_batch.unsqueeze(1),
    }[PARAM.stream_P_feature_type]

    sA_out = self.streamA_prenet(sA_inputs)
    sP_out = self.streamP_prenet(sP_inputs)

    sA_out_res = sA_out
    sP_out_res = sP_out

    for tsb in self.layers_TSB:
      sA_out, sP_out = tsb(sA_out, sP_out)

    sA_out = sA_out + sA_out_res
    sP_out = sP_out + sP_out_res

    sA_out = self.streamA_postnet(sA_out)
    sP_out_normed_stft, sP_out_angle = self.streamP_postnet(sP_out)

    est_mask = sA_out
    normed_complex_phase = sP_out_normed_stft
    est_angle = sP_out_angle
    return NET_NID_OUT(mag_mask=est_mask,
                          normalized_complex_phase=normed_complex_phase,
                          angle=est_angle)


class Losses(
    collections.namedtuple("Losses",
                           ("sum_loss", "show_losses", "stop_criterion_loss"))):
  pass


class NID(nn.Module):
  def __init__(self, mode, device):
    super(NID, self).__init__()
    self.mode = mode
    self.device = device
    self._net_model = NetNID()
    self._stft_fn = conv_stft.ConvSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length)
    self._istft_fn = conv_stft.ConviSTFT(PARAM.frame_length, PARAM.frame_step, PARAM.fft_length)

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      self.to(self.device)
      return

    self._global_step = 1
    self._start_epoch = 1
    self._nan_grads_batch = 0

    if PARAM.optimizer == "Adam":
      self._optimizer = torch.optim.Adam(self.parameters(), lr=PARAM.learning_rate)
    elif PARAM.optimizer == "RMSProp":
      self._optimizer = torch.optim.RMSprop(self.parameters(), lr=PARAM.learning_rate)

    self._lr_scheduler = None
    if PARAM.use_lr_warmup:
      def warmup(step):
        return misc_utils.warmup_coef(step, warmup_steps=PARAM.warmup_steps)
      self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, warmup)

    self.to(self.device)

  def save_every_epoch(self, ckpt_path):
    self._start_epoch += 1
    torch.save({
                "global_step": self._global_step,
                "start_epoch": self._start_epoch,
                "nan_grads_batch": self._nan_grads_batch,
                "other_state": self.state_dict(),
            }, ckpt_path)

  def load(self, ckpt_path):
    ckpt = torch.load(ckpt_path)
    self._global_step = ckpt["global_step"]
    self._start_epoch = ckpt["start_epoch"]
    self._nan_grads_batch = ckpt["nan_grads_batch"]
    self.load_state_dict(ckpt["other_state"])

  def update_params(self, loss):
    self.zero_grad()
    loss.backward()
    has_nan_inf = 0
    for params in self.parameters():
      if params.requires_grad:
        has_nan_inf += torch.sum(torch.isnan(params.grad))
        has_nan_inf += torch.sum(torch.isinf(params.grad))

    if has_nan_inf == 0:
      self._optimizer.step()
      self._lr_scheduler.step(self._global_step)
      self._global_step += 1
      return
    self._nan_grads_batch += 1

  def __call__(self, mixed_wav_batch):
    mixed_wav_batch = mixed_wav_batch.to(self.device)
    mixed_stft_batch = self._stft_fn(mixed_wav_batch)
    mixed_stft_real = mixed_stft_batch[:, 0, :, :]
    mixed_stft_imag = mixed_stft_batch[:, 1, :, :]
    mixed_mag_batch = torch.sqrt(mixed_stft_real**2+mixed_stft_imag**2)
    mixed_angle_batch = torch.atan2(mixed_stft_imag, mixed_stft_real)
    _N, _F, _T = mixed_mag_batch.size()
    mixed_normed_stft_batch = torch.div(mixed_stft_batch, mixed_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)
    self.mixed_wav_features = WavFeatures(wav_batch=mixed_wav_batch,
                                          stft_batch=mixed_stft_batch,
                                          mag_batch=mixed_mag_batch,
                                          angle_batch=mixed_angle_batch,
                                          normed_stft_batch=mixed_normed_stft_batch)

    feature_in = self.mixed_wav_features

    net_nid_out = self._net_model(feature_in)
    est_clean_angle_batch = net_nid_out.angle
    est_clean_mag_batch = torch.mul(self.mixed_wav_features.mag_batch, net_nid_out.mag_mask)

    est_clean_stft_batch = torch.cat((est_clean_mag_batch.unsqueeze(1)*torch.cos(est_clean_angle_batch.unsqueeze(1)), 
                                      est_clean_mag_batch.unsqueeze(1)*torch.sin(est_clean_angle_batch.unsqueeze(1))), dim=1)
    _N, _F, _T = est_clean_mag_batch.size()
    est_normed_stft_batch = torch.div(est_clean_stft_batch, est_clean_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)
    _mixed_wav_length = mixed_wav_batch.size()[-1]
    est_clean_wav_batch = self._istft_fn(est_clean_stft_batch, _mixed_wav_length)
    _mixed_wav_length = self.mixed_wav_features.wav_batch.size()[-1]
    est_clean_wav_batch = est_clean_wav_batch[:, :_mixed_wav_length]

    return WavFeatures(wav_batch=est_clean_wav_batch,
                       stft_batch=est_clean_stft_batch,
                       mag_batch=est_clean_mag_batch,
                       angle_batch=est_clean_angle_batch,
                       normed_stft_batch=est_normed_stft_batch)

  def get_losses(self, est_wav_features:WavFeatures, clean_wav_batch):
    self.clean_wav_batch = clean_wav_batch.to(self.device)
    self.clean_stft_batch = self._stft_fn(self.clean_wav_batch)
    clean_stft_real = self.clean_stft_batch[:, 0, :, :]
    clean_stft_imag = self.clean_stft_batch[:, 1, :, :]
    self.clean_mag_batch = torch.sqrt(clean_stft_real**2+clean_stft_imag**2)
    self.clean_angle_batch = torch.atan2(clean_stft_imag, clean_stft_real)
    _N, _F, _T = self.clean_mag_batch.size()
    self.clean_normed_stft_batch = torch.div(
        self.clean_stft_batch, self.clean_mag_batch.view(_N, 1, _F, _T)+PARAM.stft_div_norm_eps)

    est_clean_mag_batch = est_wav_features.mag_batch
    est_clean_stft_batch = est_wav_features.stft_batch
    est_clean_wav_batch = est_wav_features.wav_batch
    est_clean_normed_stft_batch = est_wav_features.normed_stft_batch

    all_losses = list()
    all_losses.extend(PARAM.sum_losses)
    all_losses.extend(PARAM.show_losses)
    all_losses.extend(PARAM.stop_criterion_losses)
    all_losses = set(all_losses)

    self.loss_compressedMag_mse = 0
    self.loss_compressedStft_mse = 0
    self.loss_mag_mse = 0
    self.loss_mag_reMse = 0
    self.loss_stft_mse = 0
    self.loss_stft_reMse = 0
    self.loss_mag_mae = 0
    self.loss_mag_reMae = 0
    self.loss_stft_mae = 0
    self.loss_stft_reMae = 0
    self.loss_wav_L1 = 0
    self.loss_wav_L2 = 0
    self.loss_wav_reL2 = 0
    self.loss_CosSim = 0
    self.loss_SquareCosSim = 0

    if "loss_compressedMag_mse" in all_losses:
      self.loss_compressedMag_mse = losses.batchSum_compressedMag_mse(
          est_clean_mag_batch, self.clean_mag_batch, PARAM.loss_compressedMag_idx)
    if "loss_compressedStft_mse" in all_losses:
      self.loss_compressedStft_mse = losses.batchSum_compressedStft_mse(
          est_clean_mag_batch, est_clean_normed_stft_batch,
          self.clean_mag_batch, self.clean_normed_stft_batch,
          PARAM.loss_compressedMag_idx)

    if "loss_mag_mse" in all_losses:
      self.loss_mag_mse = losses.batchSum_MSE(est_clean_mag_batch, self.clean_mag_batch)
    if "loss_mag_reMse" in all_losses:
      self.loss_mag_reMse = losses.batchSum_relativeMSE(est_clean_mag_batch, self.clean_mag_batch,
                                                        PARAM.relative_loss_epsilon, PARAM.RL_idx)
    if "loss_stft_mse" in all_losses:
      self.loss_stft_mse = losses.batchSum_MSE(est_clean_stft_batch, self.clean_stft_batch)
    if "loss_stft_reMse" in all_losses:
      self.loss_stft_reMse = losses.batchSum_relativeMSE(est_clean_stft_batch, self.clean_stft_batch,
                                                         PARAM.relative_loss_epsilon, PARAM.RL_idx)


    if "loss_mag_mae" in all_losses:
      self.loss_mag_mae = losses.batchSum_MAE(est_clean_mag_batch, self.clean_mag_batch)
    if "loss_mag_reMae" in all_losses:
      self.loss_mag_reMae = losses.batchSum_relativeMAE(est_clean_mag_batch, self.clean_mag_batch,
                                                        PARAM.relative_loss_epsilon)
    if "loss_stft_mae" in all_losses:
      self.loss_stft_mae = losses.batchSum_MAE(est_clean_stft_batch, self.clean_stft_batch)
    if "loss_stft_reMae" in all_losses:
      self.loss_stft_reMae = losses.batchSum_relativeMAE(est_clean_stft_batch, self.clean_stft_batch,
                                                         PARAM.relative_loss_epsilon)


    if "loss_wav_L1" in all_losses:
      self.loss_wav_L1 = losses.batchSum_MAE(est_clean_wav_batch, self.clean_wav_batch)
    if "loss_wav_L2" in all_losses:
      self.loss_wav_L2 = losses.batchSum_MSE(est_clean_wav_batch, self.clean_wav_batch)
    if "loss_wav_reL2" in all_losses:
      self.loss_wav_reL2 = losses.batchSum_relativeMSE(est_clean_wav_batch, self.clean_wav_batch,
                                                       PARAM.relative_loss_epsilon, PARAM.RL_idx)

    if "loss_CosSim" in all_losses:
      self.loss_CosSim = losses.batchMean_CosSim_loss(est_clean_wav_batch, self.clean_wav_batch)
    if "loss_SquareCosSim" in all_losses:
      self.loss_SquareCosSim = losses.batchMean_SquareCosSim_loss(
          est_clean_wav_batch, self.clean_wav_batch)
    loss_dict = {
        'loss_compressedMag_mse': self.loss_compressedMag_mse,
        'loss_compressedStft_mse': self.loss_compressedStft_mse,
        'loss_mag_mse': self.loss_mag_mse,
        'loss_mag_reMse': self.loss_mag_reMse,
        'loss_stft_mse': self.loss_stft_mse,
        'loss_stft_reMse': self.loss_stft_reMse,
        'loss_mag_mae': self.loss_mag_mae,
        'loss_mag_reMae': self.loss_mag_reMae,
        'loss_stft_mae': self.loss_stft_mae,
        'loss_stft_reMae': self.loss_stft_reMae,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_wav_L2': self.loss_wav_L2,
        'loss_wav_reL2': self.loss_wav_reL2,
        'loss_CosSim': self.loss_CosSim,
        'loss_SquareCosSim': self.loss_SquareCosSim,
    }
    sum_loss = 0.0
    sum_loss_names = PARAM.sum_losses
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_w) > 0:
        loss_t = loss_t * PARAM.sum_losses_w[i]
      sum_loss += loss_t
    show_losses = []
    show_loss_names = PARAM.show_losses
    for i, name in enumerate(show_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.show_losses_w) > 0:
        loss_t = loss_t * PARAM.show_losses_w[i]
      show_losses.append(loss_t)
    show_losses = torch.stack(show_losses)
    stop_criterion_losses_sum = 0.0
    stop_criterion_loss_names = PARAM.stop_criterion_losses
    for i, name in enumerate(stop_criterion_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.stop_criterion_losses_w) > 0:
        loss_t = loss_t * PARAM.stop_criterion_losses_w[i]
      stop_criterion_losses_sum += loss_t

    return Losses(sum_loss=sum_loss,
                  show_losses=show_losses,
                  stop_criterion_loss=stop_criterion_losses_sum)

  @property
  def global_step(self):
    return self._global_step

  @property
  def start_epoch(self):
    return self._start_epoch

  @property
  def nan_grads_batch(self):
    return self._nan_grads_batch

  @property
  def optimizer_lr(self):
    return self._optimizer.param_groups[0]['lr']

if __name__ == "__main__":
    from thop import profile
    device = 'cpu'
    test_model = NetNID()
    wav_feature_in = WavFeatures(wav_batch=torch.rand(1, PARAM.sampling_rate).to(device),
                                 stft_batch=torch.rand(1, 2, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step).to(device),
                                 mag_batch=torch.rand(1, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step).to(device),
                                 angle_batch=torch.rand(1, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step).to(device),
                                 normed_stft_batch=torch.rand(1, 2, PARAM.frequency_dim, PARAM.sampling_rate//PARAM.frame_step).to(device))
    macs, params = profile(test_model, inputs=(wav_feature_in,))
    print("Config class name: %s\n"%PARAM().config_name())
    print("MACCs of processing 1s wav = %.2fG\n"%(macs/1e9))
    print("params = %.2fM\n\n\n"%(params/1e6))
