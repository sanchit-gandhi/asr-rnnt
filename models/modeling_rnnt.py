from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Any

import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules import RNNTJoint
from nemo.collections.asr.metrics.rnnt_wer_bpe import RNNTBPEWER, RNNTBPEDecoding, RNNTBPEDecodingConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers.utils import ModelOutput
from nemo.utils import logging
from nemo.collections.asr.metrics.wer import move_dimension_to_the_front
import editdistance

class RNNTBPEWERCustom(RNNTBPEWER):
    def __init__(
            self,
            decoding: RNNTBPEDecoding,
            batch_dim_index=0,
            use_cer: bool = False,
            log_prediction: bool = True,
            dist_sync_on_step=False,
    ):
        super(RNNTBPEWERCustom, self).__init__(decoding, batch_dim_index, use_cer, log_prediction, dist_sync_on_step)
        self.add_state("cer_scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("chars", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ):
        words = 0.0
        chars = 0.0
        wer_scores = 0.0
        cer_scores = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            targets_cpu_tensor = move_dimension_to_the_front(targets_cpu_tensor, self.batch_dim_index)
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                references.append(reference)

            hypotheses, _ = self.decoding.rnnt_decoder_predictions_tensor(encoder_output, encoded_lengths)

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference :{references[0]}")
            logging.info(f"predicted :{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            h_list = list(h)
            r_list = list(r)
            h_cer_list = h.split()
            r_cer_list = r.split()
            words += len(r_list)
            chars += len(r_cer_list)
            # Compute Levenshtein's distance
            wer_scores += editdistance.eval(h_list, r_list)
            cer_scores += editdistance.eval(h_cer_list, r_cer_list)

        del hypotheses

        self.scores += torch.tensor(wer_scores, device=self.scores.device, dtype=self.scores.dtype)
        self.words += torch.tensor(words, device=self.words.device, dtype=self.words.dtype)
        self.chars += torch.tensor(chars, device=self.chars.device, dtype=self.chars.dtype)
        self.cer_scores += torch.tensor(cer_scores, device=self.cer_scores.device, dtype=self.cer_scores.dtype)

    def accumulate(self):
        return self.scores.detach(), self.words.detach(), self.cer_scores.detach(), self.chars.detach()


class RNNTJointCustom(RNNTJoint):
    def forward(
            self,
            encoder_outputs: torch.Tensor,
            decoder_outputs: Optional[torch.Tensor],
            encoder_lengths: Optional[torch.Tensor] = None,
            transcripts: Optional[torch.Tensor] = None,
            transcript_lengths: Optional[torch.Tensor] = None,
            compute_wer: bool = False,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError(
                    "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
                    "decoder_outputs can only be None for fused step!"
                )

            out = self.joint(encoder_outputs, decoder_outputs)  # [B, T, U, V + 1]
            return out

        else:
            # At least the loss module must be supplied during fused joint
            if self._loss is None or self._wer is None:
                raise ValueError("`fuse_loss_wer` flag is set, but `loss` and `wer` modules were not provided! ")

            # If fused joint step is required, fused batch size is required as well
            if self._fused_batch_size is None:
                raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

            # When using fused joint step, both encoder and transcript lengths must be provided
            if (encoder_lengths is None) or (transcript_lengths is None):
                raise ValueError(
                    "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
                )

            losses = []
            batch_size = int(encoder_outputs.size(0))  # actual batch size

            # Iterate over batch using fused_batch_size steps
            for batch_idx in range(0, batch_size, self._fused_batch_size):
                begin = batch_idx
                end = min(begin + self._fused_batch_size, batch_size)

                # Extract the sub batch inputs
                # sub_enc = encoder_outputs[begin:end, ...]
                # sub_transcripts = transcripts[begin:end, ...]
                sub_enc = encoder_outputs.narrow(dim=0, start=begin, length=end - begin)
                sub_transcripts = transcripts.narrow(dim=0, start=begin, length=end - begin)

                sub_enc_lens = encoder_lengths[begin:end]
                sub_transcript_lens = transcript_lengths[begin:end]

                # Sub transcripts does not need the full padding of the entire batch
                # Therefore reduce the decoder time steps to match
                max_sub_enc_length = sub_enc_lens.max()
                max_sub_transcript_length = sub_transcript_lens.max()

                if decoder_outputs is not None:
                    # Reduce encoder length to preserve computation
                    # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
                    if sub_enc.shape[1] != max_sub_enc_length:
                        sub_enc = sub_enc.narrow(dim=1, start=0, length=max_sub_enc_length)

                    # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
                    sub_dec = decoder_outputs.narrow(dim=0, start=begin, length=end - begin)  # [sub-batch, U, D]

                    # Reduce decoder length to preserve computation
                    # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
                    if sub_dec.shape[1] != max_sub_transcript_length + 1:
                        sub_dec = sub_dec.narrow(dim=1, start=0, length=max_sub_transcript_length + 1)

                    # Perform joint => [sub-batch, T', U', V + 1]
                    sub_joint = self.joint(sub_enc, sub_dec)

                    del sub_dec

                    # Reduce transcript length to correct alignment
                    # Transcript: [sub-batch, L] -> [sub-batch, L']; L' <= L
                    if sub_transcripts.shape[1] != max_sub_transcript_length:
                        sub_transcripts = sub_transcripts.narrow(dim=1, start=0, length=max_sub_transcript_length)

                    # Compute sub batch loss
                    # preserve loss reduction type
                    loss_reduction = self.loss.reduction

                    # override loss reduction to sum
                    self.loss.reduction = None

                    # compute and preserve loss
                    loss_batch = self.loss(
                        log_probs=sub_joint,
                        targets=sub_transcripts,
                        input_lengths=sub_enc_lens,
                        target_lengths=sub_transcript_lens,
                    )
                    losses.append(loss_batch)

                    # reset loss reduction type
                    self.loss.reduction = loss_reduction

                else:
                    losses = None

                # Update WER for sub batch
                if compute_wer:
                    sub_enc = sub_enc.transpose(1, 2)  # [B, T, D] -> [B, D, T]
                    sub_enc = sub_enc.detach()
                    sub_transcripts = sub_transcripts.detach()

                    # Update WER on each process without syncing
                    self.wer.update(sub_enc, sub_enc_lens, sub_transcripts, sub_transcript_lens)

                del sub_enc, sub_transcripts, sub_enc_lens, sub_transcript_lens

            # Collect sub batch loss results
            if losses is not None:
                losses = torch.cat(losses, 0)
                losses = losses.mean()  # global batch size average
            else:
                losses = None

            # Collect sub batch wer results
            if compute_wer:
                # Sync and all_reduce on all processes, compute global WER
                wer_num, wer_denom, cer_num, cer_denom = self.wer.accumulate()
                self.wer.reset()
            else:
                wer_num = wer_denom = cer_num = cer_denom = None

            return losses, wer_num, wer_denom, cer_num, cer_denom

@dataclass
class RNNTOutput(ModelOutput):
    """
    Base class for RNNT outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    wer_num: Optional[float] = None
    wer_denom: Optional[float] = None
    cer_num: Optional[float] = None
    cer_denom: Optional[float] = None


# Adapted from https://github.com/NVIDIA/NeMo/blob/66c7677cd4a68d78965d4905dd1febbf5385dff3/nemo/collections/asr/models/rnnt_bpe_models.py#L33
class RNNTBPEModel(EncDecRNNTBPEModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg, trainer=None)
        # Setup wer object
        self.wer = RNNTBPEWERCustom(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )
        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_wer(self.wer)

    def encoding(
            self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the acoustic model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `forward` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    def forward(self, input_ids, input_lengths=None, labels=None, label_lengths=None):
        # encoding() only performs encoder forward
        encoded, encoded_len = self.encoding(input_signal=input_ids, input_signal_length=input_lengths)
        del input_ids

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=labels, target_length=label_lengths)

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=labels, input_lengths=encoded_len, target_lengths=target_length
            )
            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)
            wer_num = wer_denom = cer_num = cer_denom = None
            if not self.training:
                self.wer.update(encoded, encoded_len, labels, target_length)
                wer_num, wer_denom, cer_num, cer_denom = self.wer.accumulate()
                self.wer.reset()

        else:
            # If experimental fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer_num, wer_denom, cer_num, cer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=labels,
                transcript_lengths=label_lengths,
                compute_wer=not self.training,
            )
            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

        return RNNTOutput(loss=loss_value, wer_num=wer_num, wer_denom=wer_denom, cer_num=cer_num, cer_denom=cer_denom)

    def change_decoding_strategy(self, decoding_cfg: DictConfig):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
        )

        self.wer = RNNTBPEWERCustom(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")
