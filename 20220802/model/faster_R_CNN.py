from torch import Tensor
from torch import nn
from torch.jit.annotations import Dict
from torchvision import ops
from torchvision.ops import MultiScaleRoIAlign

from config.train_config import cfg
from model.backbone.mobilenet import *
from model.backbone.resnet50_fpn_model import *
from utils.anchor_utils import AnchorsGenerator
from utils.faster_rcnn_utils import FastRCNNPredictor
from utils.faster_rcnn_utils import TwoMLPHead
from utils.roi_header_util import RoIHeads
from utils.rpn_utils import RPNHead, RegionProposalNetwork
from utils.transform_utils import GeneralizedRCNNTransform


def create_model(num_classes, is_relation=False):
    global backbone, model
    backbone_network = cfg.backbone

    anchor_sizes = tuple((f,) for f in cfg.anchor_size)
    aspect_ratios = tuple((f,) for f in cfg.anchor_ratio) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    if backbone_network == 'mobilenet':
        backbone = MobileNetV2(weights_path=cfg.backbone_pretrained_weights).features
        backbone.out_channels = 1280

        roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],  # roi pooling in which resolution feature
                                            output_size=cfg.roi_out_size,  # roi_pooling output feature size
                                            sampling_ratio=cfg.roi_sample_rate)  # sampling_ratio

        model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                           # transform parameters
                           min_size=cfg.min_size, max_size=cfg.max_size,
                           image_mean=cfg.image_mean, image_std=cfg.image_std,
                           # rpn parameters
                           rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                           rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
                           rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
                           rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
                           rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
                           rpn_nms_thresh=cfg.rpn_nms_thresh,
                           rpn_fg_iou_thresh=cfg.rpn_fg_iou_thresh,
                           rpn_bg_iou_thresh=cfg.rpn_bg_iou_thresh,
                           rpn_batch_size_per_image=cfg.rpn_batch_size_per_image,
                           rpn_positive_fraction=cfg.rpn_positive_fraction,
                           # Box parameters
                           box_head=None, box_predictor=None,

                           # remove low threshold target
                           box_score_thresh=cfg.box_score_thresh,
                           box_nms_thresh=cfg.box_nms_thresh,
                           box_detections_per_img=cfg.box_detections_per_img,
                           box_fg_iou_thresh=cfg.box_fg_iou_thresh,
                           box_bg_iou_thresh=cfg.box_bg_iou_thresh,
                           box_batch_size_per_image=cfg.box_batch_size_per_image,
                           box_positive_fraction=cfg.box_positive_fraction,
                           bbox_reg_weights=cfg.bbox_reg_weights
                           )
    elif backbone_network == 'resnet50_fpn':
        backbone = resnet50_fpn_backbone()

        roi_pooler = ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=cfg.roi_out_size,
            sampling_ratio=cfg.roi_sample_rate)
        model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                           # transform parameters
                           min_size=cfg.min_size, max_size=cfg.max_size,
                           image_mean=cfg.image_mean, image_std=cfg.image_std,
                           # rpn parameters
                           rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                           rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
                           rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
                           rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
                           rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
                           rpn_nms_thresh=cfg.rpn_nms_thresh,
                           rpn_fg_iou_thresh=cfg.rpn_fg_iou_thresh,
                           rpn_bg_iou_thresh=cfg.rpn_bg_iou_thresh,
                           rpn_batch_size_per_image=cfg.rpn_batch_size_per_image,
                           rpn_positive_fraction=cfg.rpn_positive_fraction,
                           # Box parameters
                           box_head=None, box_predictor=None,

                           # remove low threshold target
                           box_score_thresh=cfg.box_score_thresh,
                           box_nms_thresh=cfg.box_nms_thresh,
                           box_detections_per_img=cfg.box_detections_per_img,
                           box_fg_iou_thresh=cfg.box_fg_iou_thresh,
                           box_bg_iou_thresh=cfg.box_bg_iou_thresh,
                           box_batch_size_per_image=cfg.box_batch_size_per_image,
                           box_positive_fraction=cfg.box_positive_fraction,
                           bbox_reg_weights=cfg.bbox_reg_weights
                           )

        # weights_dict = torch.load(cfg.pretrained_weights)
        # missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        # if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        #     print("missing_keys: ", missing_keys)
        #     print("unexpected_keys: ", unexpected_keys)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return self.eager_outputs(losses, detections)


class FasterRCNN(FasterRCNNBase):
    """
    Implementation of Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or inference mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    :param backbone: (nn.Module), the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
    :param num_classes: (int), number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
    :param min_size: (int), minimum size of the image to be rescaled before feeding it to the backbone
    :param max_size: (int), maximum size of the image to be rescaled before feeding it to the backbone
    :param image_mean: (Tuple[float, float, float]):, mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
    :param image_std: (Tuple[float, float, float]), std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
    :param rpn_anchor_generator: (AnchorGenerator), module that generates the anchors for a set of feature maps.
    :param rpn_head: (nn.Module),  module that computes the objectness and regression deltas from the RPN
    :param rpn_pre_nms_top_n_train:(int),  number of proposals to keep before applying NMS during training
    :param rpn_pre_nms_top_n_test: (int), number of proposals to keep before applying NMS during testing
    :param rpn_post_nms_top_n_train: (int), number of proposals to keep after applying NMS during training
    :param rpn_post_nms_top_n_test: (int), number of proposals to keep after applying NMS during testing
    :param rpn_nms_thresh: (float), NMS threshold used for postprocessing the RPN proposals
    :param rpn_fg_iou_thresh:(float), minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
    :param rpn_bg_iou_thresh:(float), maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
    :param rpn_batch_size_per_image: (int), number of anchors that are sampled during training of the RPN
            for computing the loss
    :param rpn_positive_fraction: (float), proportion of positive anchors in a mini-batch during training
            of the RPN
    :param box_roi_pool:(MultiScaleRoIAlign), the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
    :param box_head:(nn.Module), module that takes the cropped feature maps as input
    :param box_predictor:(nn.Module), module that takes the output of box_head and returns the
            classification logits and box regression deltas.
    :param box_score_thresh:(float),during inference, only return proposals with a classification score
            greater than box_score_thresh
    :param box_nms_thresh: (float), NMS threshold for the prediction head. Used during inference
    :param box_detections_per_img: (int), maximum number of detections per image, for all classes.
    :param box_fg_iou_thresh:(float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
    :param box_bg_iou_thresh: (float), maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
    :param box_batch_size_per_image: (int), number of proposals that are sampled during training of the
            classification head
    :param box_positive_fraction: (float), proportion of positive proposals in a mini-batch during training
            of the classification head
    :param bbox_reg_weights: (Tuple[float, float, float, float]), weights for the encoding/decoding of the
            bounding boxes
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=300, max_size=800,  # preprocess minimum and maximum size
                 image_mean=None, image_std=None,  # mean and std in preprocess

                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # kept proposals before nms
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # kept proposals after nms
                 rpn_nms_thresh=0.7,  # iou threshold during nms
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # bg/fg threshold
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # number of samples and fraction

                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,

                 # remove low threshold target
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None
                 ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # output channels of the backbone
        out_channels = backbone.out_channels

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # two fc layer after roi pooling
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # get prediction
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class Relation_FasterRCNN(FasterRCNNBase):
    pass
