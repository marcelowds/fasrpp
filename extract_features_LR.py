import net
import torch
import os
import numpy as np
import PIL
from PIL import Image
from collections import defaultdict

adaface_models = {
    'ir_18': "pretrained_adaface/adaface_ir18_casia.ckpt"
}


def load_pretrained_model(architecture='ir_18'):
    assert architecture in adaface_models.keys()

    model = net.build_model(architecture)

    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {
        key[6:]: val
        for key, val in statedict.items()
        if key.startswith('model.')
    }

    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([bgr_img.transpose(2, 0, 1)]).float()
    return tensor


def extract_features_by_identity(model, image_root, device):

    features_by_identity = defaultdict(list)

    image_count = 0

    for identity_dir in sorted(os.listdir(image_root)):

        identity_path = os.path.join(image_root, identity_dir)

        if not os.path.isdir(identity_path):
            continue

        for img_name in sorted(os.listdir(identity_path)):

            img_path = os.path.join(identity_path, img_name)

            if not os.path.isfile(img_path):
                continue

            #with PIL.Image.open(img_path) as img:
            with Image.open(img_path) as img:

                aligned_rgb_img = img.resize((112, 112))
                bgr_tensor_input = to_input(aligned_rgb_img).to(device)

            with torch.no_grad():
                feature_gl, _ = model(bgr_tensor_input)

            features_by_identity[identity_dir].append(feature_gl.cpu())

            image_count += 1

            if image_count % 50 == 0:
                print(f">> Processadas {image_count} imagens... (ID atual: {identity_dir})")

    return features_by_identity


if __name__ == '__main__':

    image_root = "./sample_imgs/LR_imgs"
    output_file = "./sample_imgs/features/extracted_LR.t"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = load_pretrained_model('ir_18').to(device)

    print("Extraindo features...")

    features_by_identity = extract_features_by_identity(
        model,
        image_root,
        device
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    torch.save(features_by_identity, output_file)

    print(f"\nFeatures salvas em:")
    print(output_file)

    print(f"\nTotal de identidades: {len(features_by_identity)}")

    total_imgs = sum(len(v) for v in features_by_identity.values())
    print(f"Total de imagens: {total_imgs}")
