import os
import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################
# Delta Feature Combiner
######################################################
class FeatureCombinerCasia(nn.Module):
    def __init__(self, dropout=0.2, eta=1.0):
        super().__init__()

        self.eta = eta

        self.fc = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 512, bias=False),
        )

    def forward(self, f1, f2):

        x = torch.cat([f1, f2], dim=1)

        delta = self.fc(x)

        avg = (f1 + f2) / 2

        out = avg + self.eta * delta

        out = F.normalize(out, p=2, dim=1)

        return out


######################################################
# Combine all features from one identity
######################################################
def combine_identity_features(features, combiner, device):
    """
    Combina todas as features de uma identidade.

    Parameters
    ----------
    features : list
        Lista de tensores (1,512) ou (512,).

    Returns
    -------
    torch.Tensor
        Feature final (512,)
    """

    # Remove batch dimension e garante CPU
    features = [
        (feat.squeeze(0) if feat.dim() == 2 else feat).cpu()
        for feat in features
    ]

    if len(features) == 0:
        raise RuntimeError("Identity without features.")

    if len(features) == 1:
        return features[0]

    combined_features = []

    i = 0

    while i < len(features) - 1:

        f1 = features[i].unsqueeze(0).to(device)
        f2 = features[i + 1].unsqueeze(0).to(device)

        with torch.no_grad():
            combined = combiner(f1, f2)

        combined_features.append(combined.squeeze(0).cpu())

        i += 2

    # Número ímpar de features
    if i == len(features) - 1:
        combined_features.append(features[-1])

    if len(combined_features) == 1:
        return combined_features[0]

    return torch.stack(combined_features).mean(dim=0)


######################################################
# Main
######################################################
if __name__ == "__main__":

    input_features = "./sample_imgs/features/extracted_LR.t"
    delta_model = "./delta_pretrained/best_params_delta.pt"
    output_features = "./sample_imgs/features/combined_NN_features.t"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    ##################################################
    # Load extracted features
    ##################################################
    print("Loading extracted features...")

    features_dict = torch.load(
        input_features,
        map_location="cpu"
    )

    ##################################################
    # Load pretrained Delta network
    ##################################################
    combiner = FeatureCombinerCasia().to(device)

    state_dict = torch.load(
        delta_model,
        map_location=device
    )

    combiner.load_state_dict(state_dict)
    combiner.eval()

    print("Delta network loaded successfully.")

    ##################################################
    # Sort identities
    ##################################################
    identities = sorted(
        features_dict.keys(),
        key=lambda x: int(x)
    )

    ##################################################
    # Combine features
    ##################################################
    final_features = []

    for identity in identities:

        feature = combine_identity_features(
            features_dict[identity],
            combiner,
            device
        )

        final_features.append(feature)

    final_tensor = torch.stack(final_features)

    ##################################################
    # Save
    ##################################################
    os.makedirs(
        os.path.dirname(output_features),
        exist_ok=True
    )

    torch.save(final_tensor, output_features)

    print(f"Final tensor shape: {final_tensor.shape}")
    print(f"Combined features saved to: {output_features}")
