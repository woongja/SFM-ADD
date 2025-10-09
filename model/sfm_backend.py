import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq

from model.aasist_backend import AASIST_Backend
from model.conformertcm_backend import ConformerTCM_Backend

############################
## FOR fine-tuned SSL MODEL
############################


class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = '/home/woongjae/wildspoof/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

class SFM_ADD(nn.Module):
    def __init__(self, device, fusion="learnable"):
        super(SFM_ADD, self).__init__()
        self.xlsr = SSLModel(device)   # 1개만
        self.aasist = AASIST_Backend(input_dim=1024)
        self.conformer = ConformerTCM_Backend(input_dim=1024, emb_size=256)
        self.fusion = fusion
        if fusion == "learnable":
            self.fusion_layer = nn.Linear(1024, 1)

    def forward(self, x, labels=None):
        x_emb = self.xlsr.extract_feat(x.squeeze(-1))  # 공유
        score_a = self.aasist(x_emb)   # logits [B,2]
        score_c = self.conformer(x_emb)

        # 3. Fusion
        if self.fusion == "sum":
            logits = (score_a + score_c) / 2
        elif self.fusion == "learnable":
            alpha = torch.sigmoid(self.fusion_layer(x_emb.mean(dim=1)))  # [B,1]
            logits = alpha * score_a + (1 - alpha) * score_c
        else:
            logits = (score_a + score_c) / 2  # fallback

        out = {"logits": logits, "score_a": score_a, "score_c": score_c}

        if labels is not None:
            loss_a = F.cross_entropy(score_a, labels)
            loss_c = F.cross_entropy(score_c, labels)
            out["loss"] = 0.5*loss_a + 0.5*loss_c
            out["loss_a"] = loss_a
            out["loss_c"] = loss_c

        return out


class Model(nn.Module):
    """
    Wrapper class for training script compatibility.
    main.py 에서 importlib 로 불러올 때 항상 'Model' 클래스를 찾으므로
    여기에 SFM_ADD를 감싸서 리턴.
    """
    def __init__(self, args, device):
        super().__init__()
        fusion = args.get("fusion", "sum") if isinstance(args, dict) else "sum"
        self.model = SFM_ADD(device=device, fusion=fusion)

    def forward(self, x, labels=None):
        return self.model(x, labels)