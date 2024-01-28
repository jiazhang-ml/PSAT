import numpy as np
import tensorflow

def pro_loss1(y_true, y_pred):
    # loss of prototype
    batch = 4
    r = 2
    prototype_loss_batch = 0
    for batch_num in range(batch):
        for iiiii in range(2):
            eu_distance1 = -1 * tensorflow.norm(y_pred[batch_num, ::] - y_true[iiiii, ::])  # Negative Euclidean distance
            eu_distance1 = eu_distance1 / r
            gussian_distance1 = tensorflow.exp(eu_distance1)

            eu_distance2 = -1 * tensorflow.norm(y_pred[batch_num, ::] - y_true[iiiii, ::])  # Negative Euclidean distance
            gussian_distance2 = tensorflow.exp(eu_distance2)
            prototype_loss = tensorflow.cond(y_true[2,batch_num] == iiiii,
                             lambda:-tensorflow.compat.v1.log(gussian_distance2) / 2,
                             lambda:-tensorflow.compat.v1.log(1-gussian_distance1) / 2)
            prototype_loss_batch = prototype_loss_batch + prototype_loss
    return prototype_loss_batch



