import torch
import torch.nn as nn
from PLM_head import MLP, SpeciesAdapter
from gnn_module import build_species_encoder_from_graph
from fusion_modules import build_fusion
# from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoModel, AutoTokenizer, AlbertTokenizer


# Allowed values for ESM2(species_mode=...).
# - 'none'    : no species channel (peptide_emb only).
# - 'adapter' : legacy non-linear adapter on a pre-computed 768-d species
#               vector (original baseline in this repo).
# - 'gnn'     : taxonomy GNN channel, returns a per-batch [B, gnn_out_dim] vec.
# - 'both'    : both 'adapter' and 'gnn' channels concatenated with peptide_emb.
SPECIES_MODES = ("none", "adapter", "gnn", "both")


"""
Check your path of pre-trained models
"""
#### from huggingface directly
cache_dir = "/NAS/luyq/huggingface/hub/"
protbert_checkpoint = "Rostlab/prot_bert"   # "Rostlab/prot_bert_bfd"
amplify_120m_checkpoint = "chandar-lab/AMPLIFY_120M"
amplify_120m_base_checkpoint = "chandar-lab/AMPLIFY_120M_base"
amplify_350m_checkpoint = "chandar-lab/AMPLIFY_350M"
esm2_650m_checkpoint = "facebook/esm2_t33_650M_UR50D"

#### local (downloaded from huggingface website)
protalbert_checkpoint = "/NAS/luyq/huggingface/hub/prot_albert"
esm2_8m_checkpoint = "/NAS/luyq/huggingface/hub/esm2_t6_8M_UR50D"
esm2_35m_checkpoint = "/NAS/luyq/huggingface/hub/esm2_t12_35M_UR50D"
esm2_150m_checkpoint = "/NAS/luyq/huggingface/hub/esm2_t30_150M_UR50D"
esm2_650m_checkpoint = "/NAS/luyq/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
esm2_3b_checkpoint = "/NAS/luyq/esm2_models/esm2_t36_3B_UR50D"


'''
Tokenizer
'''
def load_tokenizer(plm_type):
    if plm_type == "tape":
        tokenizer = TAPETokenizer(vocab='iupac')
    elif plm_type in ['protbert', 'protalbert']:
        if plm_type == 'protbert':
            tokenizer = AutoTokenizer.from_pretrained(protbert_checkpoint, cache_dir=cache_dir,
                                                    do_lower_case=False)
        elif plm_type == 'protalbert':
            tokenizer = AlbertTokenizer.from_pretrained(protalbert_checkpoint,
                                                    do_lower_case=False)
    elif "esm2" in plm_type:
        if plm_type.split('-')[-1] == '8M':
            tokenizer = AutoTokenizer.from_pretrained(esm2_8m_checkpoint)
        elif plm_type.split('-')[-1] == '35M':      # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_35m_checkpoint)
        elif plm_type.split('-')[-1] == '150M':     # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_150m_checkpoint)
        elif plm_type.split('-')[-1] == '650M':     # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_650m_checkpoint, cache_dir=cache_dir)
        elif plm_type.split('-')[-1] == '3B':       # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_3b_checkpoint)
    elif "AMPLIFY" in plm_type:
        if plm_type == 'AMPLIFY-120M':
            tokenizer = AutoTokenizer.from_pretrained(amplify_120m_checkpoint, cache_dir=cache_dir)
        elif plm_type == 'AMPLIFY-120M-base':
            tokenizer = AutoTokenizer.from_pretrained(amplify_120m_base_checkpoint, cache_dir=cache_dir)
        elif plm_type == 'AMPLIFY-350M':
            tokenizer = AutoTokenizer.from_pretrained(amplify_350m_checkpoint, cache_dir=cache_dir)
    return tokenizer


'''
TAPE model
'''
class TAPE(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(TAPE, self).__init__()
        self.tape = ProteinBertModel.from_pretrained('bert-base')
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        
        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.tape.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(768, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(768, [1024, 512, 128, 32, 2])  ## 5layers

    def forward(self, inputs):
        if self.plm_output == 'mean':  # mean of sequence_output
            outputs = self.tape(inputs)[
                0]  # [batch_size, , seq_len,  hidden_size]
            outputs = torch.mean(outputs, dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':  # output of every sample's <cls> token
            outputs = self.tape(
                inputs, input_mask=None)[0][:, 0]  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        return outputs.view(-1, outputs.size(-1))


'''
ProtBert model
'''
class ProtBert(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(ProtBert, self).__init__()
        self.proteinbert = AutoModel.from_pretrained(protbert_checkpoint, cache_dir=cache_dir)

        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.proteinbert.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(1024, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(1024, [1024, 512, 128, 32, 2])  ## 5layers
        self.plm_output = plm_output

    def forward(self, input_ids):
        outputs = self.proteinbert(input_ids)

        if self.plm_output == 'mean':
            outputs = outputs[0].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs[0][:, 0]
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))


'''
ProtAlBert model
'''
class ProtAlBert(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(ProtAlBert, self).__init__()
        self.proteinbert = AutoModel.from_pretrained(protalbert_checkpoint)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.proteinbert.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(4096, [256, 64, 2])               ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(4096, [1024, 512, 128, 32, 2])    ## 5layers
        self.plm_output = plm_output

    def forward(self, input_ids):
        outputs = self.proteinbert(input_ids)

        if self.plm_output == 'mean':
            outputs = outputs[0].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs[0][:, 0]
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))


'''
ESM2 family
'''
class ESM2(nn.Module):
    """ESM2 + optional species side-channel.

    Backwards-compatible wrt. ``use_species`` (legacy bool flag for the
    PubMedBERT-adapter baseline). The new ``species_mode`` argument supersedes
    it and supports four configurations (see ``SPECIES_MODES``).
    """

    def __init__(self,
                 head_type='3MLP',
                 plm_output='mean',
                 finetune_plm=True,
                 esm_size='8M',
                 # ----- legacy adapter knobs (kept for compatibility) -----
                 use_species=False,
                 species_in_dim=768,
                 species_out_dim=128,
                 species_bottleneck=128,
                 species_dropout=0.1,
                 # ----- new species-mode dispatcher -----
                 species_mode=None,
                 # ----- GNN species-channel knobs -----
                 taxo_graph_path=None,
                 gnn_hidden=128,
                 gnn_out_dim=64,
                 gnn_layers=2,
                 gnn_type='gcn',
                 gnn_heads=4,
                 gnn_dropout=0.1,
                 gnn_fusion='leaf',
                 gnn_hier_levels=('species', 'genus', 'family'),
                 gnn_freeze_init=True,
                 use_lora_init=False,
                 lora_rank=16,
                 gnn_residual=False,
                 gnn_layernorm=False,
                 # ----- ESM<->GNN fusion strategy (Stage 3) -----
                 fusion_strategy='concat'):
        super(ESM2, self).__init__()
        if esm_size == '8M':
            self.checkpoint = esm2_8m_checkpoint
            self.hidden_size = 320
        elif esm_size == '35M':
            self.checkpoint = esm2_35m_checkpoint
            self.hidden_size = 480
        elif esm_size == '150M':
            self.checkpoint = esm2_150m_checkpoint
            self.hidden_size = 640
        elif esm_size == '650M':
            self.checkpoint = esm2_650m_checkpoint
            self.hidden_size = 1280
        elif esm_size == '3B':
            self.checkpoint = esm2_3b_checkpoint
            self.hidden_size = 2560
        else:
            raise ValueError(f"Wrong size of ESM2: {esm_size}")
        self.esm = AutoModel.from_pretrained(self.checkpoint, cache_dir=cache_dir)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Resolve species_mode. If unset, fall back to the legacy bool so that
        # existing CLI invocations (--use_species) still work unchanged.
        if species_mode is None:
            species_mode = "adapter" if use_species else "none"
        if species_mode not in SPECIES_MODES:
            raise ValueError(f"species_mode must be one of {SPECIES_MODES}, got {species_mode!r}")
        self.species_mode = species_mode
        self.use_species = species_mode != "none"   # kept for backward compat
        print(self.plm_output, self.hidden_size, "species_mode=", species_mode)

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.esm.parameters():
                param.requires_grad = False

        # ----- species channels --------------------------------------------------
        # 'adapter': legacy 768-d -> species_out_dim non-linear adapter.
        if species_mode in ("adapter", "both"):
            self.species_adapter = SpeciesAdapter(
                in_dim=species_in_dim,
                bottleneck=species_bottleneck,
                out_dim=species_out_dim,
                dropout=species_dropout,
            )
        else:
            self.species_adapter = None

        # 'gnn': taxonomy-DAG GNN -> [B, gnn_out_dim].
        if species_mode in ("gnn", "both"):
            if taxo_graph_path is None:
                raise ValueError(
                    "species_mode='%s' requires taxo_graph_path to be provided "
                    "(produced by scripts/build_taxonomy_graph.py)." % species_mode
                )
            self.species_gnn = build_species_encoder_from_graph(
                taxo_graph_path,
                hidden=gnn_hidden,
                out_dim=gnn_out_dim,
                num_layers=gnn_layers,
                gnn_type=gnn_type,
                heads=gnn_heads,
                dropout=gnn_dropout,
                fusion=gnn_fusion,
                hier_levels=gnn_hier_levels,
                freeze_init=gnn_freeze_init,
                use_lora_init=use_lora_init,
                lora_rank=lora_rank,
                use_residual=gnn_residual,
                use_layernorm=gnn_layernorm,
            )
        else:
            self.species_gnn = None
        # cached for forward ergonomics
        self.gnn_out_dim = gnn_out_dim
        self.species_out_dim = species_out_dim
        self.fusion_strategy = fusion_strategy

        # ----- ESM<->GNN fusion module (Stage 3) -----
        # Only relevant when species_mode includes 'gnn'. For adapter-only or
        # 'both', the adapter still uses simple concat (legacy behaviour).
        if species_mode in ("gnn", "both") and fusion_strategy != "concat":
            self.fusion = build_fusion(
                fusion_strategy,
                esm_dim=self.hidden_size,
                gnn_out_dim=gnn_out_dim,
                dropout=gnn_dropout,
            )
        else:
            self.fusion = None

        # ----- compute the input dim of the regression head ---------------------
        if species_mode in ("gnn", "both") and self.fusion is not None:
            # Non-concat fusions output a fixed-width vector (= esm_dim).
            mlp_in = self.fusion.out_dim
            if species_mode == "both":
                mlp_in += species_out_dim
        else:
            mlp_in = self.hidden_size
            if species_mode in ("adapter", "both"):
                mlp_in += species_out_dim
            if species_mode in ("gnn", "both"):
                mlp_in += gnn_out_dim

        # Last layer has NO activation (the MLP class breaks before adding one)
        # so the scalar output is unrestricted -- essential for log-MIC values
        # that can be negative.
        if head_type == '3MLP':
            self.projection = MLP(mlp_in, [256, 64, 1])
        elif head_type == '5MLP':
            self.projection = MLP(mlp_in, [1024, 512, 128, 32, 1])
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, input_ids, species_emb=None, species_ids=None, species_names=None):
        """
        input_ids     : LongTensor [B, L]
        species_emb   : FloatTensor [B, species_in_dim] -- 'adapter' / 'both' modes
        species_ids   : LongTensor [B] of species rows in the GNN graph,
                        OR a Python list of species name strings -- 'gnn' / 'both' modes
        species_names : alias for ``species_ids`` for readability when passing names.
        """
        # cross_attn fusion needs token-level output; for the rest, mean/cls is enough
        need_tokens = (self.fusion is not None) and (self.fusion_strategy == "cross_attn")
        outputs = self.esm(input_ids)
        seq_tokens = outputs[0] if need_tokens else None         # [B, L, hidden_size]
        if self.plm_output == 'mean':
            seq_rep = outputs[0].mean(dim=1)        # [B, hidden_size]
        elif self.plm_output == 'cls':
            seq_rep = outputs[0][:, 0]              # [B, hidden_size]
        else:
            raise ValueError(f"Unknown plm_output: {self.plm_output}")

        # ----- adapter branch (legacy) -----
        adapter_emb = None
        if self.species_mode in ("adapter", "both"):
            if species_emb is None:
                raise ValueError(f"species_mode={self.species_mode!r} but species_emb is None.")
            adapter_emb = self.species_adapter(species_emb)             # [B, species_out_dim]

        # ----- GNN branch -----
        gnn_emb = None
        if self.species_mode in ("gnn", "both"):
            sp_in = species_ids if species_ids is not None else species_names
            if sp_in is None:
                raise ValueError(
                    f"species_mode={self.species_mode!r} but neither species_ids nor "
                    f"species_names was provided."
                )
            gnn_emb = self.species_gnn(sp_in)                           # [B, gnn_out_dim]

        # ----- fuse ESM + GNN -----
        if gnn_emb is not None and self.fusion is not None:
            seq_fused = self.fusion(seq_rep, gnn_emb, seq_tokens=seq_tokens)
            if adapter_emb is not None:
                fused = torch.cat([seq_fused, adapter_emb], dim=-1)
            else:
                fused = seq_fused
        else:
            feats = [seq_rep]
            if adapter_emb is not None:
                feats.append(adapter_emb)
            if gnn_emb is not None:
                feats.append(gnn_emb)
            fused = feats[0] if len(feats) == 1 else torch.cat(feats, dim=-1)

        out = self.projection(fused)
        return out.view(-1, out.size(-1))


'''
AMPLIFY family
'''
class AMPLIFY(nn.Module): 
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True, amplify_type='AMPLIFY-120M'):
        super(AMPLIFY, self).__init__()
        if amplify_type in ['AMPLIFY-120M', 'AMPLIFY-120M-base']:
            self.hidden_size = 640
        elif amplify_type in ['AMPLIFY-350M', 'AMPLIFY-350M-base']:
            self.hidden_size = 960
        else:
            raise ValueError(f"Wrong type of AMPLIFY: {amplify_type}")
        self.encoder = AutoModel.from_pretrained(
            f"chandar-lab/{amplify_type.replace('-', '_')}", trust_remote_code=True, cache_dir=cache_dir)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        print(self.plm_output, self.hidden_size)

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        # print(f"Trainable encoder parameters: {trainable_params:,}")
        
        if head_type == '3MLP':
            self.projection = MLP(self.hidden_size, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(self.hidden_size, [1024, 512, 128, 32, 2])  ## 5layers

    def forward(self, input_ids):
        outputs = self.encoder(input_ids, output_hidden_states=True)
        if self.plm_output == 'mean':
            outputs = outputs.hidden_states[-1].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs.hidden_states[-1][:, 0]
            # print(outputs.shape) #!
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))

