import numpy as np
def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()
    mask_gt = target.cpu().numpy()
    mask_seg = bin_input.cpu().numpy()
    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)
    #print(2 * np.sum(np.logical_and(mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg)))

    return score.mean()




def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / all_positives

    return recall.mean()


def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / all_positive_calls

    return precision.mean()