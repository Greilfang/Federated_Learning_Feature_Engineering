import torch


def encrypt_feature_in_rand(feature, tr_name):
    mask_tensor = torch.rand_like(feature)
    if tr_name in ["sum", "subtract"]:
        encrypted_feature = mask_tensor + feature
    else:
        encrypted_feature = mask_tensor * feature
    return encrypted_feature, mask_tensor


def decrypt_feature_in_rand(feature, mask_tensor, tr_name):
    if tr_name in ["sum", "subtract"]:
        primordial_feature = feature - mask_tensor
    else:
        primordial_feature = feature / mask_tensor
    return primordial_feature