import torch

import utils
from utils.hparams import hparams
from .diff.net import DiffNet
from .diff.shallow_diffusion_tts import GaussianDiffusion, OfflineGaussianDiffusion
from .diffspeech_task import DiffSpeechTask
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from modules.fastspeech.pe import PitchExtractor
from modules.fastspeech.fs2 import FastSpeech2
from modules.diffsinger_midi.fs2 import FastSpeech2MIDI
from modules.fastspeech.tts_modules import mel2ph_to_dur

from usr.diff.candidate_decoder import FFT
from utils.pitch_utils import denorm_f0
from tasks.tts.fs2_utils import FastSpeechDataset
from tasks.tts.fs2 import FastSpeech2Task

import numpy as np
import os
import torch.nn.functional as F

DIFF_DECODERS = {
    "wavenet": lambda hp: DiffNet(hp["audio_num_mel_bins"]),
    "fft": lambda hp: FFT(
        hp["hidden_size"], hp["dec_layers"], hp["dec_ffn_kernel_size"], hp["num_heads"]
    ),
}


class DiffSingerTask(DiffSpeechTask):
    def __init__(self):
        super(DiffSingerTask, self).__init__()
        self.dataset_cls = FastSpeechDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        if hparams.get("pe_enable") is not None and hparams["pe_enable"]:
            self.pe = PitchExtractor().cuda()
            utils.load_ckpt(self.pe, hparams["pe_ckpt"], "model", strict=True)
            self.pe.eval()

    def build_tts_model(self):
        # import torch
        # from tqdm import tqdm
        # v_min = torch.ones([80]) * 100
        # v_max = torch.ones([80]) * -100
        # for i, ds in enumerate(tqdm(self.dataset_cls('train'))):
        #     v_max = torch.max(torch.max(ds['mel'].reshape(-1, 80), 0)[0], v_max)
        #     v_min = torch.min(torch.min(ds['mel'].reshape(-1, 80), 0)[0], v_min)
        #     if i % 100 == 0:
        #         print(i, v_min, v_max)
        # print('final', v_min, v_max)
        mel_bins = hparams["audio_num_mel_bins"]
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins,
            denoise_fn=DIFF_DECODERS[hparams["diff_decoder_type"]](hparams),
            timesteps=hparams["timesteps"],
            K_step=hparams["K_step"],
            loss_type=hparams["diff_loss_type"],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )
        if hparams["fs2_ckpt"] != "":
            utils.load_ckpt(self.model.fs2, hparams["fs2_ckpt"], "model", strict=True)
            # self.model.fs2.decoder = None
            # for k, v in self.model.fs2.named_parameters():
            #    v.requires_grad = False

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample["txt_tokens"]  # [B, T_t]

        target = sample["mels"]  # [B, T_s, 80]
        energy = sample["energy"]
        # fs2_mel = sample['fs2_mels']
        spk_embed = (
            sample.get("spk_embed")
            if not hparams["use_spk_id"]
            else sample.get("spk_ids")
        )
        mel2ph = sample["mel2ph"]
        f0 = sample["f0"]
        uv = sample["uv"]

        outputs["losses"] = {}

        outputs["losses"], model_out = self.run_model(
            self.model, sample, return_output=True, infer=False
        )

        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams["num_valid_plots"]:
            model_out = self.model(
                txt_tokens,
                spk_embed=spk_embed,
                mel2ph=mel2ph,
                f0=f0,
                uv=uv,
                energy=energy,
                ref_mels=None,
                infer=True,
            )

            if hparams.get("pe_enable") is not None and hparams["pe_enable"]:
                gt_f0 = self.pe(sample["mels"])[
                    "f0_denorm_pred"
                ]  # pe predict from GT mel
                pred_f0 = self.pe(model_out["mel_out"])[
                    "f0_denorm_pred"
                ]  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample["f0"], sample["uv"], hparams)
                pred_f0 = model_out.get("f0_denorm")
            self.plot_wav(
                batch_idx,
                sample["mels"],
                model_out["mel_out"],
                is_mel=True,
                gt_f0=gt_f0,
                f0=pred_f0,
            )
            self.plot_mel(
                batch_idx,
                sample["mels"],
                model_out["mel_out"],
                name=f"diffmel_{batch_idx}",
            )
            self.plot_mel(
                batch_idx,
                sample["mels"],
                model_out["fs2_mel"],
                name=f"fs2mel_{batch_idx}",
            )
        return outputs


class ShallowDiffusionOfflineDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(ShallowDiffusionOfflineDataset, self).__getitem__(index)
        item = self._get_item(index)

        if self.prefix != "train" and hparams["fs2_ckpt"] != "":
            fs2_ckpt = os.path.dirname(hparams["fs2_ckpt"])
            item_name = item["item_name"]
            fs2_mel = torch.Tensor(
                np.load(f"{fs2_ckpt}/P_mels_npy/{item_name}.npy")
            )  # ~M generated by FFT-singer.
            sample["fs2_mel"] = fs2_mel
        return sample

    def collater(self, samples):
        batch = super(ShallowDiffusionOfflineDataset, self).collater(samples)
        if self.prefix != "train" and hparams["fs2_ckpt"] != "":
            batch["fs2_mels"] = utils.collate_2d([s["fs2_mel"] for s in samples], 0.0)
        return batch


class DiffSingerOfflineTask(DiffSingerTask):
    def __init__(self):
        super(DiffSingerOfflineTask, self).__init__()
        self.dataset_cls = ShallowDiffusionOfflineDataset

    def build_tts_model(self):
        mel_bins = hparams["audio_num_mel_bins"]
        self.model = OfflineGaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins,
            denoise_fn=DIFF_DECODERS[hparams["diff_decoder_type"]](hparams),
            timesteps=hparams["timesteps"],
            K_step=hparams["K_step"],
            loss_type=hparams["diff_loss_type"],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )
        # if hparams['fs2_ckpt'] != '':
        #     utils.load_ckpt(self.model.fs2, hparams['fs2_ckpt'], 'model', strict=True)
        #     self.model.fs2.decoder = None

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        target = sample["mels"]  # [B, T_s, 80]
        mel2ph = sample["mel2ph"]  # [B, T_s]
        f0 = sample["f0"]
        uv = sample["uv"]
        energy = sample["energy"]
        fs2_mel = None  # sample['fs2_mels']
        spk_embed = (
            sample.get("spk_embed")
            if not hparams["use_spk_id"]
            else sample.get("spk_ids")
        )
        if hparams["pitch_type"] == "cwt":
            cwt_spec = sample[f"cwt_spec"]
            f0_mean = sample["f0_mean"]
            f0_std = sample["f0_std"]
            sample["f0_cwt"] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        output = model(
            txt_tokens,
            mel2ph=mel2ph,
            spk_embed=spk_embed,
            ref_mels=[target, fs2_mel],
            f0=f0,
            uv=uv,
            energy=energy,
            infer=infer,
        )

        losses = {}
        if "diff_loss" in output:
            losses["mel"] = output["diff_loss"]
        # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        # if hparams['use_pitch_embed']:
        #     self.add_pitch_loss(output, sample, losses)
        if hparams["use_energy_embed"]:
            self.add_energy_loss(output["energy_pred"], energy, losses)

        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample["txt_tokens"]  # [B, T_t]

        target = sample["mels"]  # [B, T_s, 80]
        energy = sample["energy"]
        # fs2_mel = sample['fs2_mels']
        spk_embed = (
            sample.get("spk_embed")
            if not hparams["use_spk_id"]
            else sample.get("spk_ids")
        )
        mel2ph = sample["mel2ph"]
        f0 = sample["f0"]
        uv = sample["uv"]

        outputs["losses"] = {}

        outputs["losses"], model_out = self.run_model(
            self.model, sample, return_output=True, infer=False
        )

        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < hparams["num_valid_plots"]:
            fs2_mel = sample["fs2_mels"]
            model_out = self.model(
                txt_tokens,
                spk_embed=spk_embed,
                mel2ph=mel2ph,
                f0=f0,
                uv=uv,
                energy=energy,
                ref_mels=[None, fs2_mel],
                infer=True,
            )
            if hparams.get("pe_enable") is not None and hparams["pe_enable"]:
                gt_f0 = self.pe(sample["mels"])[
                    "f0_denorm_pred"
                ]  # pe predict from GT mel
                pred_f0 = self.pe(model_out["mel_out"])[
                    "f0_denorm_pred"
                ]  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample["f0"], sample["uv"], hparams)
                pred_f0 = model_out.get("f0_denorm")
            self.plot_wav(
                batch_idx,
                sample["mels"],
                model_out["mel_out"],
                is_mel=True,
                gt_f0=gt_f0,
                f0=pred_f0,
            )
            self.plot_mel(
                batch_idx,
                sample["mels"],
                model_out["mel_out"],
                name=f"diffmel_{batch_idx}",
            )
            self.plot_mel(
                batch_idx, sample["mels"], fs2_mel, name=f"fs2mel_{batch_idx}"
            )
        return outputs

    def test_step(self, sample, batch_idx):
        spk_embed = (
            sample.get("spk_embed")
            if not hparams["use_spk_id"]
            else sample.get("spk_ids")
        )
        txt_tokens = sample["txt_tokens"]
        energy = sample["energy"]
        if hparams["profile_infer"]:
            pass
        else:
            mel2ph, uv, f0 = None, None, None
            if hparams["use_gt_dur"]:
                mel2ph = sample["mel2ph"]
            if hparams["use_gt_f0"]:
                f0 = sample["f0"]
                uv = sample["uv"]
            fs2_mel = sample["fs2_mels"]
            outputs = self.model(
                txt_tokens,
                spk_embed=spk_embed,
                mel2ph=mel2ph,
                f0=f0,
                uv=uv,
                ref_mels=[None, fs2_mel],
                energy=energy,
                infer=True,
            )
            sample["outputs"] = self.model.out2mel(outputs["mel_out"])
            sample["mel2ph_pred"] = outputs["mel2ph"]

            if hparams.get("pe_enable") is not None and hparams["pe_enable"]:
                sample["f0"] = self.pe(sample["mels"])[
                    "f0_denorm_pred"
                ]  # pe predict from GT mel
                sample["f0_pred"] = self.pe(sample["outputs"])[
                    "f0_denorm_pred"
                ]  # pe predict from Pred mel
            else:
                sample["f0"] = denorm_f0(sample["f0"], sample["uv"], hparams)
                sample["f0_pred"] = outputs.get("f0_denorm")
            return self.after_infer(sample)


class MIDIDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(MIDIDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample["f0_midi"] = torch.FloatTensor(item["f0_midi"])
        sample["pitch_midi"] = torch.LongTensor(item["pitch_midi"])[
            : hparams["max_frames"]
        ]

        return sample

    def collater(self, samples):
        batch = super(MIDIDataset, self).collater(samples)
        batch["f0_midi"] = utils.collate_1d([s["f0_midi"] for s in samples], 0.0)
        batch["pitch_midi"] = utils.collate_1d([s["pitch_midi"] for s in samples], 0)
        # print((batch['pitch_midi'] == f0_to_coarse(batch['f0_midi'])).all())
        return batch


class M4SingerDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(M4SingerDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample["pitch_midi"] = torch.LongTensor(item["pitch_midi"])
        sample["midi_dur"] = torch.FloatTensor(item["midi_dur"])
        sample["is_slur"] = torch.LongTensor(item["is_slur"])
        sample["word_boundary"] = torch.LongTensor(item["word_boundary"])
        sample["lang"] = torch.LongTensor(item["lang"])
        sample["speechsing"] = torch.LongTensor(item["speechsing"])
        return sample

    def collater(self, samples):
        batch = super(M4SingerDataset, self).collater(samples)
        batch["pitch_midi"] = utils.collate_1d([s["pitch_midi"] for s in samples], 0)
        batch["midi_dur"] = utils.collate_1d([s["midi_dur"] for s in samples], 0)
        batch["is_slur"] = utils.collate_1d([s["is_slur"] for s in samples], 0)
        batch["word_boundary"] = utils.collate_1d(
            [s["word_boundary"] for s in samples], 0
        )
        batch["lang"] = utils.collate_1d([s["lang"] for s in samples], 0)
        batch["speechsing"] = torch.LongTensor([s["speechsing"] for s in samples])
        return batch


class DiffSingerMIDITask(DiffSingerTask):
    def __init__(self):
        super(DiffSingerMIDITask, self).__init__()
        # self.dataset_cls = MIDIDataset
        self.dataset_cls = M4SingerDataset

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        target = sample["mels"]  # [B, T_s, 80]
        # mel2ph = sample['mel2ph'] if hparams['use_gt_dur'] else None # [B, T_s]
        mel2ph = sample["mel2ph"]
        if (
            hparams.get("switch_midi2f0_step") is not None
            and self.global_step > hparams["switch_midi2f0_step"]
        ):
            f0 = None
            uv = None
        else:
            f0 = sample["f0"]
            uv = sample["uv"]
        energy = sample["energy"]

        spk_embed = (
            sample.get("spk_embed")
            if not hparams["use_spk_id"]
            else sample.get("spk_ids")
        )
        if hparams["pitch_type"] == "cwt":
            cwt_spec = sample[f"cwt_spec"]
            f0_mean = sample["f0_mean"]
            f0_std = sample["f0_std"]
            sample["f0_cwt"] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        output = model(
            txt_tokens,
            mel2ph=mel2ph,
            spk_embed=spk_embed,
            ref_mels=target,
            f0=f0,
            uv=uv,
            energy=energy,
            infer=infer,
            pitch_midi=sample["pitch_midi"],
            midi_dur=sample.get("midi_dur"),
            is_slur=sample.get("is_slur"),
            lang=sample.get("lang"),
            speechsing=sample.get("speechsing"),
        )

        losses = {}
        if "diff_loss" in output:
            losses["mel"] = output["diff_loss"]
        self.add_dur_loss(
            output["dur"], mel2ph, txt_tokens, sample["word_boundary"], losses=losses
        )
        if hparams["use_pitch_embed"]:
            self.add_pitch_loss(output, sample, losses)
        if hparams["use_energy_embed"]:
            self.add_energy_loss(output["energy_pred"], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample["txt_tokens"]  # [B, T_t]

        target = sample["mels"]  # [B, T_s, 80]
        energy = sample["energy"]
        # fs2_mel = sample['fs2_mels']
        spk_embed = (
            sample.get("spk_embed")
            if not hparams["use_spk_id"]
            else sample.get("spk_ids")
        )
        mel2ph = sample["mel2ph"]

        outputs["losses"] = {}

        outputs["losses"], model_out = self.run_model(
            self.model, sample, return_output=True, infer=False
        )

        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx % 20 == 0 and batch_idx // 20 < hparams["num_valid_plots"]:
            model_out = self.model(
                txt_tokens,
                spk_embed=spk_embed,
                mel2ph=mel2ph,
                f0=None,
                uv=None,
                energy=energy,
                ref_mels=None,
                infer=True,
                pitch_midi=sample["pitch_midi"],
                midi_dur=sample.get("midi_dur"),
                is_slur=sample.get("is_slur"),
                lang=sample.get("lang"),
                speechsing=sample.get("speechsing"),
            )

            if hparams.get("pe_enable") is not None and hparams["pe_enable"]:
                gt_f0 = self.pe(sample["mels"])[
                    "f0_denorm_pred"
                ]  # pe predict from GT mel
                pred_f0 = self.pe(model_out["mel_out"])[
                    "f0_denorm_pred"
                ]  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample["f0"], sample["uv"], hparams)
                pred_f0 = model_out.get("f0_denorm")
            self.plot_wav(
                batch_idx,
                sample["mels"],
                model_out["mel_out"],
                is_mel=True,
                gt_f0=gt_f0,
                f0=pred_f0,
            )
            self.plot_mel(
                batch_idx,
                sample["mels"],
                model_out["mel_out"],
                name=f"diffmel_{batch_idx}",
            )
            self.plot_mel(
                batch_idx,
                sample["mels"],
                model_out["fs2_mel"],
                name=f"fs2mel_{batch_idx}",
            )
            if hparams["use_pitch_embed"]:
                self.plot_pitch(batch_idx, sample, model_out)
        return outputs

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, wdb, losses=None):
        """
        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        else:
            raise NotImplementedError

        # use linear scale for sent and word duration
        if hparams["lambda_word_dur"] > 0:
            idx = F.pad(wdb.cumsum(axis=1), (1, 0))[:, :-1]
            # word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_(1, idx, midi_dur)  # midi_dur can be implied by add gt-ph_dur
            word_dur_p = dur_pred.new_zeros([B, idx.max() + 1]).scatter_add(
                1, idx, dur_pred
            )
            word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_add(
                1, idx, dur_gt
            )
            wdur_loss = F.mse_loss(
                (word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none"
            )
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * hparams["lambda_word_dur"]
        if hparams["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss(
                (sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean"
            )
            losses["sdur"] = sdur_loss.mean() * hparams["lambda_sent_dur"]


class AuxDecoderMIDITask(FastSpeech2Task):
    def __init__(self):
        super().__init__()
        # self.dataset_cls = MIDIDataset
        self.dataset_cls = M4SingerDataset

    def build_tts_model(self):
        if hparams.get("use_midi") is not None and hparams["use_midi"]:
            self.model = FastSpeech2MIDI(self.phone_encoder)
        else:
            self.model = FastSpeech2(self.phone_encoder)

    def run_model(self, model, sample, return_output=False):
        txt_tokens = sample["txt_tokens"]  # [B, T_t]
        target = sample["mels"]  # [B, T_s, 80]
        mel2ph = sample["mel2ph"]  # [B, T_s]
        f0 = sample["f0"]
        uv = sample["uv"]
        energy = sample["energy"]

        spk_embed = (
            sample.get("spk_embed")
            if not hparams["use_spk_id"]
            else sample.get("spk_ids")
        )
        if hparams["pitch_type"] == "cwt":
            cwt_spec = sample[f"cwt_spec"]
            f0_mean = sample["f0_mean"]
            f0_std = sample["f0_std"]
            sample["f0_cwt"] = f0 = model.cwt2f0_norm(cwt_spec, f0_mean, f0_std, mel2ph)

        output = model(
            txt_tokens,
            mel2ph=mel2ph,
            spk_embed=spk_embed,
            ref_mels=target,
            f0=f0,
            uv=uv,
            energy=energy,
            infer=False,
            pitch_midi=sample["pitch_midi"],
            midi_dur=sample.get("midi_dur"),
            is_slur=sample.get("is_slur"),
            lang=sample.get("lang"),
            speechsing=sample.get("speechsing"),
        )

        losses = {}
        self.add_mel_loss(output["mel_out"], target, losses)
        self.add_dur_loss(
            output["dur"], mel2ph, txt_tokens, sample["word_boundary"], losses=losses
        )
        if hparams["use_pitch_embed"]:
            self.add_pitch_loss(output, sample, losses)
        if hparams["use_energy_embed"]:
            self.add_energy_loss(output["energy_pred"], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, wdb, losses=None):
        """
        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        else:
            raise NotImplementedError

        # use linear scale for sent and word duration
        if hparams["lambda_word_dur"] > 0:
            idx = F.pad(wdb.cumsum(axis=1), (1, 0))[:, :-1]
            # word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_(1, idx, midi_dur)  # midi_dur can be implied by add gt-ph_dur
            word_dur_p = dur_pred.new_zeros([B, idx.max() + 1]).scatter_add(
                1, idx, dur_pred
            )
            word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_add(
                1, idx, dur_gt
            )
            wdur_loss = F.mse_loss(
                (word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none"
            )
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * hparams["lambda_word_dur"]
        if hparams["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss(
                (sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean"
            )
            losses["sdur"] = sdur_loss.mean() * hparams["lambda_sent_dur"]

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], model_out = self.run_model(
            self.model, sample, return_output=True
        )
        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        mel_out = self.model.out2mel(model_out["mel_out"])
        outputs = utils.tensors_to_scalars(outputs)
        # if sample['mels'].shape[0] == 1:
        #     self.add_laplace_var(mel_out, sample['mels'], outputs)
        if batch_idx < hparams["num_valid_plots"]:
            self.plot_mel(batch_idx, sample["mels"], mel_out)
            self.plot_dur(batch_idx, sample, model_out)
            if hparams["use_pitch_embed"]:
                self.plot_pitch(batch_idx, sample, model_out)
        return outputs

    # ############
    # # infer
    # ############
    # def test_step(self, sample, batch_idx):
    #     spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
    #     txt_tokens = sample['txt_tokens']
    #     mel2ph, uv, f0 = None, None, None
    #     ref_mels = None
    #     if hparams['profile_infer']:
    #         pass
    #     else:
    #         if hparams['use_gt_dur']:
    #             mel2ph = sample['mel2ph']
    #         if hparams['use_gt_f0']:
    #             f0 = sample['f0']
    #             uv = sample['uv']
    #             print('Here using gt f0!!')
    #         if hparams.get('use_midi') is not None and hparams['use_midi']:
    #             outputs = self.model(
    #                 txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True,
    #                 pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'), lang=sample.get('lang'), speechsing=sample.get('speechsing'))
    #         else:
    #             outputs = self.model(
    #                 txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True, lang=sample.get('lang'), speechsing=sample.get('speechsing'))
    #         sample['outputs'] = self.model.out2mel(outputs['mel_out'])
    #         sample['mel2ph_pred'] = outputs['mel2ph']
    #         if hparams.get('pe_enable') is not None and hparams['pe_enable']:
    #             sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
    #             sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
    #         else:
    #             sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
    #             sample['f0_pred'] = outputs.get('f0_denorm')

    #         return self.after_infer(sample)
