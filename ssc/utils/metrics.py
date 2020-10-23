import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

chunk_size = int(1e3)
def compute_confusion(pred, target, nbr_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    cm = torch.zeros([nbr_classes, nbr_classes], dtype=torch.long, device=target.device)
    for idx in range(0,pred.numel(), chunk_size):
        try:
            target_chunk = target[idx:(idx+chunk_size)]
            pred_chunk = pred[idx:(idx+chunk_size)]
        except IndexError:
            target_chunk = target[idx:]
            pred_chunk = pred[idx:]
        i = torch.stack([target_chunk,pred_chunk])
        v = torch.ones(target_chunk.size(), device = target_chunk.device, dtype = torch.long)
        cm_chunk = torch.sparse.LongTensor(i, v, (nbr_classes, nbr_classes))
        cm += cm_chunk.to_dense()

    return cm

def plot_confusion(cm, classes, normalize=True, title=None, cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float')
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, cm_sum, out=np.zeros_like(cm), where=(cm_sum!=0))
        vmin = 0
        vmax = 1.0
    else:
        vmin = None
        vmax = None

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin = vmin, vmax = vmax)
    ax.figure.colorbar(im, ax=ax)

    #Mark diagonal
    ax.plot(np.arange(len(classes)), 'k.', markersize=2)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    #Reduce font size
    fontsize = 6
    plt.setp(ax.get_xticklabels(), fontsize = fontsize)
    plt.setp(ax.get_yticklabels(), fontsize = fontsize)

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    #DEBUG
    # plt.savefig('confusion.png')
    # sys.exit()

    return fig

def compute_IoU(confusion_matrix):
    intersection = np.diag(confusion_matrix).astype('float')
    union = confusion_matrix.sum(axis=0) + confusion_matrix.sum(axis=1)
    IoU = np.divide(intersection, union, out=np.zeros_like(intersection), where=(union!=0))
    return IoU
