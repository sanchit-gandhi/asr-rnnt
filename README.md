# Fine-Tuning RNN-T Models
A [PyTorch](https://github.com/pytorch/pytorch) repository for fine-tuning NVIDIA RNN-T BPE models using Hugging Face Datasets and the Hugging Face Trainer. The sister repository for fine-tuning Seq2Seq and CTC ASR models in JAX/Flax can be found at: https://github.com/sanchit-gandhi/seq2seq-speech.

The modelling files are based very heavily on those from [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). This is a standalone repository to enable rapid prototyping and involvement with the community. The final modelling files and training script will be merged into [Transformers 🤗](https://github.com/huggingface/transformers) to be used with the rest of the open-source library. The final system weights will be made publicly available at [huggingface.co](huggingface.co) 🚀

![Transducer Model](transducer-model.png?style=centerme)

**Figure 1:** RNN-T Transducer Model[^1]

## Set-Up
First, install NVIDIA NeMo 'from source' following the instructions at: https://github.com/NVIDIA/NeMo#installation.

Then, install all packages from [requirements.txt](requirements.txt):
```
pip install -r requirements.txt
```
Depending on your operating system (e.g. Linux), you might have to install `libsndfile` using your distribution’s package manager, for example:
```
sudo apt-get install libsndfile1
```
To check CUDA, NeMo and the bitsandbytes optimiser have been installed correctly, run:
```
# check CUDA installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# check NeMo installation
python -c "from nemo.collections.asr.models import EncDecRNNTBPEModel; print('NeMo: installed successfully')"

# check bitsandbytes installation
wget https://gist.githubusercontent.com/TimDettmers/1f5188c6ee6ed69d211b7fe4e381e713/raw/4d17c3d09ccdb57e9ab7eca0171f2ace6e4d2858/check_bnb_install.py
python check_bnb_install.py
```
The only thing left to do is login to Weight and Biases for some pretty looking logs!
```
wandb login
```

## Example Usage
The configuration files (.yaml) for different system architectures can be found in the [conf]() directory. These are the relevant RNN-T conf files copied one-to-one from the NVIDIA NeMo repository (https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf). All models use 'byte-pair encoding' (BPE). In practice, BPE models give faster inference and better WERs than char-based RNN-T models.

Provided within this repository are conf files for:
* ContextNet: CNN-RNN-transducer architecture, large size (~144M), with Transducer loss and sub-word encoding ([paper](https://arxiv.org/abs/2005.03191)).
* Dummy ContextNet: 2-layer ContextNet model with reduced hidden-dimensions (~0.5MB). For prototyping and debugging.

Once a conf has been selected, training can be started by running one of the sample scripts. The number of epochs is selected to give approximately the same number of training steps for all datasets (~400k). Evaluation is performed every 80k train steps. The model weights are saves every 200k train steps.
* [run_librispeech.sh](scripts/run_librispeech.sh): Train for 12 epochs on LibriSpeech ASR 960h.
* [run_common_voice_9.sh](scripts/run_common_voice_9.sh): Train for 4 epochs on Common Voice 9.
* [run_tedlium.sh](scripts/run_tedlium.sh): Train for 24 epochs on TED-LIUM (release3).
* [run_switchboard.sh](scripts/run_switchboard.sh): Train for 14 epochs on LDC SwitchBoard.
* [run_dummy.sh](scripts/run_dummy.sh): Train on a dummy dataset using a dummy model (for prototyping and debugging)

RNN-T models are extremely memory intensive due to the cost of computing the Joint module. To achieve a reasonable batch-size during training with the full-sized models (min. 8), we employ the following memory saving strategies:
* Train in fp16 (half) precision: the fprop/bprop activations/gradients are automatically downcast/upcast through the PyTorch [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html) package. Set by passing the argument `fp16` to the HF Trainer.
* 8bit optimiser: we use the 8bit implementation of AdamW from [bitsandbytes](https://github.com/facebookresearch/bitsandbytes#using-the-8-bit-optimizers)
* Filter audio samples longer than 20s: memory usage increases exponentially with sequence length.

To improve memory usage further, one could employ gradient checkpointing, or use the "fused batch step" for the Joint module (provided by NeMo in the RNN-T model)[^2]. Both of these methods improve memory usage at the expense of compute speed.

During training, evaluation is performed using "greedy" search. Following training, the final evaluation step is performed using "beam" search. Using greedy search for the intermediate evaluation steps significantly speeds-up inference time. Using beam search on the final evaluation step yields far better WERs.

[^1]: Source: https://lorenlugosch.github.io/posts/2020/11/transducer/
[^2]: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/configs.html#joint-model