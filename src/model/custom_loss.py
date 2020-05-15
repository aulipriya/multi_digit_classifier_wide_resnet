import torch.nn as nn


class Custom_loss:
    def __init__(self):
        self.criteria = nn.CrossEntropyLoss()

    def loss(self, length_logits, digit1, digit2, digit3, digit4, digit5, length, label):
        length_loss = self.criteria(length_logits, length)
        digit1_loss = self.criteria(digit1, label[0])
        digit2_loss = self.criteria(digit2, label[1])
        digit3_loss = self.criteria(digit3, label[2])
        digit4_loss = self.criteria(digit4, label[3])
        digit5_loss = self.criteria(digit5, label[4])

        return length_loss + digit1_loss + digit2_loss + digit3_loss + digit4_loss + digit5_loss
