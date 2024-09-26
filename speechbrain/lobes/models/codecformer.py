import dac
print("descript audio codec v",dac.__version__)
import torchaudio.transforms as T
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from dac.nn.layers import Snake1d
from wavtokenizer.decoder.pretrained import WavTokenizer
from transformers import MimiModel, AutoFeatureExtractor
from typing import Optional, Tuple, List
import numpy as np

class DACWrapper():
    '''
    Wrapper model for Descript Audio Codec
    '''
    def __init__(self, input_sample_rate=8000, DAC_model_path = None, DAC_sample_rate = 16000, Freeze=True):
        '''
        input_sample_rate: defaults to 8000 as it common in speech separation
        Model Path: Please provde the model path to the DAC model, otherwise the 16KHz model will be automatically downloaded.
        DAC_sample_rate: defaults to 16000. If using a DAC model other than the 16khz model, please specify this number.
        '''
        super(DACWrapper, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.DAC_sample_rate = DAC_sample_rate
        
        if DAC_model_path == None:
            model_path = dac.utils.download(model_type="16khz")
        
        self.model = dac.DAC.load(model_path)

        self.dac_sampler = T.Resample(input_sample_rate, DAC_sample_rate)
        self.org_sampler = T.Resample(DAC_sample_rate, input_sample_rate)

        def count_all_parameters(model): return sum(p.numel() for p in model.parameters())
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')

        else:
            print(f'Model with {count_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def resample_audio(self, x, condition):
        '''
        torchaudio resample function used here only requires last dimension to be time.
        condition: "dac" to set the sampling rate to the DAC sampling rate
                   "org" to set the sampling rate back to the original sampling rate

        it sucks that i have to go to cpu for this. need to think how i can make this stay in gpu
        '''
        # get device
        device = x.device

        # Implement some checks on the input
        assert len(x.shape) == 3
        B, C, T = x.shape
        assert C == 1 #model should only be handling single channel

        # Resamples the audio from the input rate to the dac model's rate
        if condition == "dac":
            x_resamp = self.dac_sampler(x)
        elif condition == "org":
            x_resamp = self.org_sampler(x)
        
        # normalize the resampled audio, otherwise we will run into clipping issues
        x_resamp = x_resamp / torch.max(x_resamp.abs(),dim=2,keepdim=True)[0]

        return x_resamp.to(device)

    def get_encoded_features(self, x):
        '''
        x should be the torch tensor as per the data loader
        Expects x to be of dimensions [Batch, Channel, Time]
        '''
        #keep original lengths from pre resampling
        original_length = x.shape[-1]

        #make the input into the desired sampling rate
        x = self.resample_audio(x, "dac")

        #Now to perform the encoding
        x = self.model.preprocess(x, self.DAC_sample_rate)
        x_enc = self.model.encoder(x)

        return x_enc, original_length #the codec outputs a different length due to the masking
    
    def get_quantized_features(self, x):
        '''
        expects input [B, D, T] where D is the Quantized continuous representation of input
        '''

        x_qnt, codes_hat, latents_hat, commitment_loss_hat, codebook_loss_hat = self.model.quantizer(x, None)
        return x_qnt, codes_hat, latents_hat, commitment_loss_hat, codebook_loss_hat

    def get_decoded_signal(self, x, original_length):
        '''
        expects input [B, D, T] where D is the Quantized continuous representation of input
        original length is the original length of the input audio
        '''
        y_hat = self.model.decoder(x)

        # out = y_hat[...,:original_length]
        # out = self.resample_audio(out, "org")
        out = self.resample_audio(y_hat, "org")
        
        # T might have changed due to model. If so, fix it here
        if out.shape[-1] != original_length:
            T_origin = original_length
            T_est = out.shape[-1]
            
            if T_origin > T_est:
                out = F.pad(out, (0, T_origin - T_est))
            else:
                out = out[:, :, :T_origin]
        return out

class simpleSeparator2(nn.Module):
    def __init__(self, num_spks, channels, block, block_channels, activation=None):
        super(simpleSeparator2, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels #this is dependent on the dac model
        self.block = block #this should be a seq2seq model with identical input and output sizes
        self.ch_down = nn.Conv1d(channels, block_channels,1,bias=False)
        self.ch_up = nn.Conv1d(block_channels, channels,1,bias=False)
        #self.time_mix = nn.Conv1d(channels,channels,1,bias=False)
        self.masker = weight_norm(nn.Conv1d(channels, channels*num_spks, 1, bias=False))

        if not activation:
            self.activation = Snake1d(channels) #nn.Tanh() #nn.ReLU() #Snake1d(channels)
        else:
            self.activation = activation
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), Snake1d(channels) #nn.Tanh() #, Snake1d(channels)#
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self,x):
        x = self.ch_down(x)
        #[B,N,L]
        x = x.permute(0,2,1)
        #[B,L,N]
        x_b = self.block(x)
        #[B,L,N]
        x_b = x_b.permute(0,2,1)
        #[B,N,L]
        x = self.ch_up(x_b)

        B, N, L = x.shape
        masks = self.masker(x)
        
        #[B,N*num_spks,L]
        masks = masks.view(B*self.num_spks,-1,L)
        
        #[B*num_spks, N, L]
        x = self.output(masks) * self.output_gate(masks)
        x = self.activation(x)

        #[B*num_spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        
        # [B, spks, N, L]
        x = x.transpose(0,1)
        # [spks, B, N, L]

        return x

class WavTokenizerWrapper:
    '''
    Wrapper model for WavTokenizer
    '''
    def __init__(self, input_sample_rate=8000, model_config_path=None, model_ckpt_path=None, tokenizer_sample_rate=24000, Freeze=True):
        '''
        input_sample_rate: defaults to 8000 as expected file input
        model_config_path: Path to the config file for WavTokenizer
        model_ckpt_path: Path to the checkpoint file for WavTokenizer
        tokenizer_sample_rate: defaults to 24000. Specify if using a model with a different sample rate.
        '''
        super(WavTokenizerWrapper, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.tokenizer_sample_rate = tokenizer_sample_rate

        if model_config_path is None or model_ckpt_path is None:
            raise ValueError("Please provide both the model config and checkpoint paths.")

        self.model = WavTokenizer.from_pretrained0802(model_config_path, model_ckpt_path)

        self.dac_sampler = T.Resample(input_sample_rate, tokenizer_sample_rate)
        self.org_sampler = T.Resample(tokenizer_sample_rate, input_sample_rate)

        def count_all_parameters(model): return sum(p.numel() for p in model.parameters())
        def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            
            print(f'Model frozen with {count_parameters(self.model)/1000000:.2f}M trainable parameters remaining')
            print(f'Model has {count_all_parameters(self.model)/1000000:.2f}M parameters in total')
        else:
            print(f'Model with {count_all_parameters(self.model)/1000000:.2f}M trainable parameters loaded')

    def resample_audio(self, x, condition):
        '''
        Resample the audio according to the condition.
        condition: "tokenizer" to set the sampling rate to the tokenizer's rate
                   "org" to set the sampling rate back to the original rate
        '''
        device = x.device

        assert len(x.shape) == 3, "Input tensor must have 3 dimensions [Batch, Channels, Time]"
        B, C, T = x.shape
        assert C == 1, "Input tensor must be mono-channel [Batch, 1, Time]"

        if condition == "tokenizer":
            x_resamp = self.dac_sampler(x)
        elif condition == "org":
            x_resamp = self.org_sampler(x)
        else:
            raise ValueError("Unknown condition for resampling: {}".format(condition))
        
        x_resamp = x_resamp / torch.max(x_resamp.abs(), dim=2, keepdim=True)[0]

        return x_resamp.to(device)

    def get_encoded_features(self, x):
        '''
        x should be a torch tensor with dimensions [Batch, Channel, Time]
        '''
        original_length = x.shape[-1]

        # Resample the audio to the tokenizer's sample rate
        x = self.resample_audio(x, "tokenizer")
    
        # Remove channel dimensions for the audio data tensor
        x = x.squeeze(1)
        
        # Generate features and discrete codes
        bandwidth_id = torch.tensor([0]).to(x.device)
        features, _, _ = self.model.feature_extractor(x, bandwidth_id=bandwidth_id)
        return features, original_length
    
    def get_quantized_features(self, x, bandwidth_id=None):
        '''
        Expects input [B, D, T] where D is the encoded continuous representation of input.
        Returns quantized features, codes, latents, commitment loss, and codebook loss in the same format as DACWrapper.
        '''
        if bandwidth_id is None:
            bandwidth_id = torch.tensor([0]).to(x.device)

        # Ensure the tensor has 3 dimensions [Batch, Channels, Time]
        if x.ndim != 3:
            raise ValueError(f"Expected input to have 3 dimensions [Batch, Channels, Time], but got {x.ndim} dimensions.")

        # Perform the quantization directly on the encoded features
        q_res = self.model.feature_extractor.encodec.quantizer(
            x, 
            frame_rate=self.model.feature_extractor.frame_rate, 
            bandwidth=self.model.feature_extractor.bandwidths[bandwidth_id]
        )

        # Extract necessary outputs
        quantized = q_res.quantized
        codes = q_res.codes
        latents = x  # The input x itself is the latent representation after encoding
        commit_loss = q_res.penalty

        # Placeholder for codebook_loss (not directly available, could be None)
        codebook_loss = None

        # Return the outputs in the expected format
        return quantized, codes, latents, commit_loss, codebook_loss

    def get_decoded_signal(self, features, original_length):
        '''
        Decodes the features back to the audio signal.
        '''
        # Decode the features to get the waveform
        bandwidth_id = torch.tensor([0]).to(features.device)

        x = self.model.backbone(features, bandwidth_id=bandwidth_id)
        y_hat = self.model.head(x)

        # Ensure the output has three dimensions [Batch, Channels, Time] before resampling
        if y_hat.ndim == 2:
            y_hat = y_hat.unsqueeze(1)  # Add a channel dimension if it's missing

        # Resample the decoded signal back to the original sampling rate
        y_hat_resampled = self.resample_audio(y_hat, "org")

        # Ensure the output shape matches the original length
        if y_hat_resampled.shape[-1] != original_length:
            T_origin = original_length
            T_est = y_hat_resampled.shape[-1]

            if T_origin > T_est:
                y_hat_resampled = F.pad(y_hat_resampled, (0, T_origin - T_est))
            else:
                y_hat_resampled = y_hat_resampled[:, :, :T_origin]

        return y_hat_resampled

class simpleSeparatorMimi(nn.Module):
    def __init__(self, num_spks, channels, block, block_channels, activation=None):
        """
        Simple separator model for use with MimiWrapper, with the same interface as simpleSeparator2.

        Args:
            num_spks: Number of speakers to separate.
            channels: Number of channels (latent dimensions from MimiWrapper).
            block: The seq2seq model or processing block to use for the separation.
            block_channels: Number of channels to use in the intermediate processing layers.
            activation: The activation function to use (optional).
        """
        super(simpleSeparatorMimi, self).__init__()
        self.num_spks = num_spks 
        self.channels = channels  # Latent dimension from MimiWrapper
        self.block = block  # Sequence processing model (e.g., transformer or LSTM)
        # def initialize_weights(module):
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             module.bias.data.fill_(0.01)

        # self.block.apply(initialize_weights)

        self.ch_down = nn.Conv1d(channels, block_channels, 1, bias=False)  # Downsample latent dimension to block channels
        self.ch_up = nn.Conv1d(block_channels, channels, 1, bias=False)  # Upsample back to latent dimension

        self.masker = weight_norm(nn.Conv1d(channels, channels * num_spks, 1, bias=False))  # Mask layer for separation

        if not activation:
            self.activation = Snake1d(channels)  # Default activation function
        else:
            self.activation = activation

        # Gated output layers
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1), Snake1d(channels)
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Initial input latents Min/Max/Mean: {x.min().item()}, {x.max().item()}, {x.mean().item()}")
        
        x = x.float()
        x = self.ch_down(x)
        # print(f"After ch_down Min/Max/Mean: {x.min().item()}, {x.max().item()}, {x.mean().item()}")
        
        x = x.permute(0, 2, 1)
        x_b = self.block(x)
        # x_b = 0.5 * x_b
        x_b = x_b.permute(0, 2, 1)
        # print(f"After block Min/Max/Mean: {x_b.min().item()}, {x_b.max().item()}, {x_b.mean().item()}")

        x = self.ch_up(x_b)
        # print(f"After ch_up Min/Max/Mean: {x.min().item()}, {x.max().item()}, {x.mean().item()}")

        B, N, L = x.shape
        masks = self.masker(x)
        # print(f"After masker Min/Max/Mean: {masks.min().item()}, {masks.max().item()}, {masks.mean().item()}")
        
        masks = masks.view(B * self.num_spks, -1, L)
        x = self.output(masks) * self.output_gate(masks)
        # print(f"After output and gate Min/Max/Mean: {x.min().item()}, {x.max().item()}, {x.mean().item()}")

        x = self.activation(x)
        # print(f"After activation Min/Max/Mean: {x.min().item()}, {x.max().item()}, {x.mean().item()}")

        x = x.view(B, self.num_spks, N, L)
        x = x.transpose(0, 1)
        return x


class MimiWrapper(nn.Module):
    def __init__(self, input_sample_rate=8000, mimi_model_name="kyutai/mimi", Freeze=False, code_length=32):
        super(MimiWrapper, self).__init__()
        self.input_sample_rate = input_sample_rate
        self.model = MimiModel.from_pretrained(mimi_model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(mimi_model_name)
        self.mimi_sample_rate = self.feature_extractor.sampling_rate
        self.code_length = code_length  # Should match the encoder's code_length (32)

        # Initialize resamplers if needed
        if self.input_sample_rate != self.mimi_sample_rate:
            self.dac_sampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.mimi_sample_rate)
            self.org_sampler = T.Resample(orig_freq=self.mimi_sample_rate, new_freq=self.input_sample_rate)
        else:
            self.dac_sampler = None
            self.org_sampler = None

        # Freeze model parameters if specified
        if Freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def resample_audio(self, x: torch.Tensor, to_mimi: bool = True) -> torch.Tensor:
        """
        Resamples audio to Mimi's sampling rate or back to the original rate.
        
        Args:
            x (torch.Tensor): Input audio tensor of shape [Batch, Channels, Time].
            to_mimi (bool): If True, resamples to Mimi's sampling rate; otherwise, resamples back.
        
        Returns:
            torch.Tensor: Resampled audio.
        """
        if to_mimi and self.dac_sampler:
            return self.dac_sampler(x)
        elif not to_mimi and self.org_sampler:
            return self.org_sampler(x)
        return x

    def get_encoded_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Encodes input audio into continuous latent embeddings.
        
        Args:
            x (torch.Tensor): Input audio tensor of shape [Batch, Channels, Time].
        
        Returns:
            Tuple[torch.Tensor, int]: Tuple containing the continuous latents and the original audio length.
        """
        original_length = x.shape[-1]
        device = x.device

        # Resample the input to Mimi's sample rate
        x = self.resample_audio(x, to_mimi=True).to(device)

        # Extract features for each batch element using the feature extractor
        batch_size = x.shape[0]
        inputs = self.feature_extractor(
            raw_audio=x.squeeze().cpu(),  # Extract batch of audio
            sampling_rate=self.mimi_sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.model.encoder(inputs["input_values"])
            encoder_outputs = self.model.encoder_transformer(
                embeddings.transpose(1, 2), return_dict=False
            )
            embeddings = encoder_outputs[0].transpose(1, 2)
            embeddings = self.model.downsample(embeddings)
            continuous_latents = embeddings
            # codes = self.model.quantizer.encode(continuous_latents).transpose(0, 1)
            # lats = self.model.quantizer.decode(codes)
            # embeddings = self.model.upsample(lats)

            # decoder_outputs = self.model.decoder_transformer(
            #     embeddings.transpose(1, 2), return_dict=False
            # )
            # embeddings = decoder_outputs[0].transpose(1, 2)
            # decoded_audio = self.model.decoder(embeddings)
            # print(decoded_audio.shape)
            # torchaudio.save("a.wav", decoded_audio.detach().squeeze(0).cpu(), self.mimi_sample_rate)

        # print(continuous_latents.shape)
        # print(f"Encoded Latents Min/Max/Avg: {continuous_latents.min().item():.4f}, {continuous_latents.max().item():.4f}, {continuous_latents.mean().item():.4f}")
        # print(f"Encoded Codes Min/Max: {codes.min().item():.4f}, {codes.max().item():.4f}")
        return continuous_latents, original_length

    def get_decoded_signal(self, x: torch.Tensor, original_length: int) -> torch.Tensor:
        """
        expects input [B, D, T] where D is the Quantized continuous representation of input
        original length is the original length of the input audio
        """
        # print(f"Decoding Min/Max: {x.min().item():.4f}, {x.max().item():.4f}")

        with torch.no_grad(): 
            if x.shape[1] == self.code_length:
                x = x.long()
                # print(x.min(), x.max())
                # print("Decoding with quantizer")
                embeddings = self.model.quantizer.decode(x)
                # print(f"Decoding Latents Min/Max/Avg: {continuous_latents.min().item():.4f}, {continuous_latents.max().item():.4f}, {continuous_latents.mean().item():.4f}")
            else:
                embeddings = x

            embeddings = self.model.upsample(embeddings)

            decoder_outputs = self.model.decoder_transformer(
                embeddings.transpose(1, 2), return_dict=False
            )
            embeddings = decoder_outputs[0].transpose(1, 2)
            decoded_audio = self.model.decoder(embeddings)

        # decoded_audio = self.model.decode(x)["audio_values"]
        # torchaudio.save("b.wav", decoded_audio.detach().squeeze(0).cpu(), self.mimi_sample_rate)

        decoded_audio = self.resample_audio(decoded_audio, to_mimi=False)

        # Ensure the decoded audio matches the original length
        if decoded_audio.shape[-1] != original_length:
            if decoded_audio.shape[-1] > original_length:
                decoded_audio = decoded_audio[..., :original_length]
            else:
                padding = original_length - decoded_audio.shape[-1]
                decoded_audio = F.pad(decoded_audio, (0, padding))

        decoded_audio.requires_grad = True
        
        return decoded_audio

    def get_quantized_features(self, x, bandwidth_id=None):
        '''
        Expects input [B, D, T] where D is the encoded continuous representation of input.
        Returns quantized features, codes, latents, commitment loss, and codebook loss in the same format as DACWrapper.
        '''

        # Perform the quantization directly on the encoded features
        quantized = self.model.quantizer.encode(x).transpose(0, 1)
        codes = quantized
        latents = x
        commit_loss = None
        codebook_loss = None

        return quantized, codes, latents, commit_loss, codebook_loss
