import tensorflow as tf

backend = tf.keras.backend


def categorical_crossentropy_with_logits(y_true, y_pred):
    # compute cross entropy
    y_pred = backend.softmax(y_pred)
    cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)

    # compute loss
    loss = backend.mean(backend.sum(cross_entropy, axis=[1, 2]))
    return loss

def dice_loss(smooth=1e-2):
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)
        intersection = backend.sum(y_true * y_pred, axis=[1, 2])
        dice_conf = (2 * intersection + smooth) / (backend.sum(y_true, axis=[1, 2]) + backend.sum(y_pred, axis=[1, 2]) + smooth)
        return 1 - backend.mean(dice_conf)
    return loss

def dice_ce_loss(smooth=1e-2, ratio=1.0):
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)
        loss_ce = backend.mean(backend.sum(cross_entropy, axis=[1, 2]))
        #loss_ce = backend.mean(cross_entropy)

        intersection = backend.sum(y_true * y_pred, axis=[1, 2])
        dice_conf = (2 * intersection + smooth) / (backend.sum(y_true, axis=[1, 2]) + backend.sum(y_pred, axis=[1, 2]) + smooth)
        loss_dice = 1 - backend.mean(dice_conf)
        return loss_ce + ratio*loss_dice
    return loss

    
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)
        # compute ce loss
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)
        # compute weights
        weights = backend.sum(alpha * backend.pow(1 - y_pred, gamma) * y_true, axis=-1)
        return backend.mean(backend.sum(weights * cross_entropy, axis=[1, 2]))

    return loss


def miou_loss(weights=None, num_classes=2):
    if weights is not None:
        assert len(weights) == num_classes
        weights = tf.convert_to_tensor(weights)
    else:
        weights = tf.convert_to_tensor([1.] * num_classes)

    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)

        inter = y_pred * y_true
        inter = backend.sum(inter, axis=[1, 2])

        union = y_pred + y_true - (y_pred * y_true)
        union = backend.sum(union, axis=[1, 2])

        return -backend.mean((weights * inter) / (weights * union + 1e-8))

    return loss

def weight_miou_loss(weights=None, num_classes=2):
    if weights is not None:
        assert len(weights) == num_classes
        weights = tf.convert_to_tensor(weights)
    else:
        w = [1.0]*num_classes
        w[0] = 0.1
        weights = tf.convert_to_tensor(w)
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)
        inter = y_pred * y_true
        inter = backend.sum(inter, axis=[1, 2])
        union = y_pred + y_true - (y_pred * y_true)
        union = backend.sum(union, axis=[1, 2])
        return -backend.mean((weights * inter) / (weights * union + 1e-8))                                                                              
    return loss

def self_balanced_focal_loss(alpha=3, gamma=2.0):
    """
    Original by Yang Lu:

    This is an improvement of Focal Loss, which has solved the problem
    that the factor in Focal Loss failed in semantic segmentation.
    It can adaptively adjust the weights of different classes in semantic segmentation
    without introducing extra supervised information.

    :param alpha: The factor to balance different classes in semantic segmentation.
    :param gamma: The factor to balance different samples in semantic segmentation.
    :return:
    """

    def loss(y_true, y_pred):
        # cross entropy loss
        y_pred = backend.softmax(y_pred, -1)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred)

        # sample weights
        sample_weights = backend.max(backend.pow(1.0 - y_pred, gamma) * y_true, axis=-1)

        # class weights
        pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                                   axis=[1, 2], keepdims=True)
        class_weights = backend.max(backend.pow(backend.ones_like(y_true) * alpha, pixel_rate) * y_true, axis=-1)

        # final loss
        final_loss = class_weights * sample_weights * cross_entropy
        return backend.mean(backend.sum(final_loss, axis=[1, 2]))

    return loss
