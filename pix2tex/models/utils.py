import torch
import torch.nn as nn
import torch.nn.functional as F
from . import hybrid
from . import transformer
from . import swin  

class Model(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor, **kwargs):
        """
        Forward pass through the encoder and decoder.
        """
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    def beam_search(self, x: torch.Tensor, beam_width: int = 5, max_seq_len: int = 20, temperature: float = 0.25):
        """
        Beam search for sequence generation.
        """
        eos_token = self.args.eos_token
        bos_token = self.args.bos_token
        device = x.device

        # Encode the input sequence
        encoded = self.encoder(x)

        # Initialize beams
        beams = [(torch.LongTensor([bos_token]).to(device), 0)]  # (sequence, score)

        for _ in range(max_seq_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1].item() == eos_token:
                    new_beams.append((seq, score))  # Keep completed sequences
                    continue

                # Generate next token probabilities
                outputs = self.decoder.generate(
                    seq.unsqueeze(0), 1, eos_token=eos_token, context=encoded, temperature=temperature
                )
                logits = outputs[0, -1]
                probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)

                # Add new beams
                for i in range(beam_width):
                    next_token = topk_indices[i]
                    next_score = (score + topk_probs[i].item()) / (len(seq) + 1)  # Normalize by length
                    new_seq = torch.cat([seq, next_token.unsqueeze(0)])
                    new_beams.append((new_seq, next_score))

            # Sort beams and prune
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Early stopping if all beams end with EOS
            if all(seq[-1].item() == eos_token for seq, _ in beams):
                break

        # Return the best sequence
        best_seq = max(beams, key=lambda x: x[1])[0]
        return best_seq

    @torch.no_grad()
    def generate(self, x: torch.Tensor, beam_width: int = 5, temperature: float = 0.25):
        """
        Generate sequence using beam search.
        """
        return self.beam_search(x, beam_width=beam_width, temperature=temperature)

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        """
        Data parallelism for multi-GPU training.
        """
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]
        replicas = nn.parallel.replicate(self, device_ids)
        inputs = nn.parallel.scatter(x, device_ids)
        kwargs = nn.parallel.scatter(kwargs, device_ids)
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()


def get_model(args):
    """
    Initialize the model with specified encoder and decoder architectures.
    """
    if args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(args)
    elif args.encoder_structure.lower() == 'swin':
        encoder = swin.get_encoder(args)  
    else:
        raise NotImplementedError(f'Encoder structure "{args.encoder_structure}" not supported.')

    decoder = transformer.get_decoder(args)
    encoder.to(args.device)
    decoder.to(args.device)

    model = Model(encoder, decoder, args)

    # Optional: Track the model with WandB
    if args.wandb:
        import wandb
        wandb.watch(model)

    return model
