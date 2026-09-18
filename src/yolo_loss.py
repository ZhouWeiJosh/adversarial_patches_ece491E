import torch
import torch.nn as nn


class YOLOv2PersonLoss(nn.Module):

    def __init__(
        self,
        num_classes=80,
        num_anchors=5,
        target_class=0,
        attack_type="obj"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # COCO:
        # class 0 = person
        self.target_class = target_class

        self.attack_type = attack_type

    def forward(self, output):
        """
        Expected YOLO output:

            [B, 425, 13, 13]

        For COCO:

            425 = 5 * (5 + 80)

        Per anchor:

            tx
            ty
            tw
            th
            objectness
            80 class logits
        """

        batch_size = output.shape[0]

        grid_h = output.shape[2]
        grid_w = output.shape[3]

        attributes = (
            5 + self.num_classes
        )

        # --------------------------------------
        # Reshape
        # --------------------------------------

        output = output.view(
            batch_size,
            self.num_anchors,
            attributes,
            grid_h,
            grid_w
        )

        # --------------------------------------
        # Objectness
        # --------------------------------------

        objectness_logits = output[:, :, 4, :, :]

        objectness = torch.sigmoid(
            objectness_logits
        )

        # --------------------------------------
        # Class probabilities
        # --------------------------------------

        class_logits = output[
            :,
            :,
            5:5 + self.num_classes,
            :,
            :
        ]

        class_probabilities = torch.softmax(
            class_logits,
            dim=2
        )

        person_probability = (
            class_probabilities[
                :,
                :,
                self.target_class,
                :,
                :
            ]
        )

        # --------------------------------------
        # Select attack objective
        # --------------------------------------

        if self.attack_type == "obj":

            # For now, target objectness specifically
            # where YOLO believes the object is a person.
            score = (
                objectness
                *
                person_probability
            )

        elif self.attack_type == "cls":

            score = person_probability

        elif self.attack_type == "obj_cls":

            score = (
                objectness
                *
                person_probability
            )

        else:

            raise ValueError(
                "attack_type must be "
                "'obj', 'cls', or 'obj_cls'"
            )

        # --------------------------------------
        # Highest-scoring YOLO prediction
        # --------------------------------------

        max_score = score.view(
            batch_size,
            -1
        ).max(dim=1).values

        return max_score.mean()
