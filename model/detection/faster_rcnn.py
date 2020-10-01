from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torchvision.ops import misc as misc_nn_ops, roi_align
# from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.poolers import LevelMapper

from .RelationNetwork import RelationNetwork
from ..utils import load_state_dict_from_url

from .generalized_rcnn import GeneralizedRCNN
from .rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from .roi_heads import RoIHeads
from .transform import GeneralizedRCNNTransform
from .backbone_utils import resnet_fpn_backbone

__all__ = [
    "FasterRCNN", "fasterrcnn_resnet50_fpn",
]


class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    Example::
        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(self, backbone, device, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
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

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size,
                relation_layers_num=1)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics present in the FPN paper.

    Arguments:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign

    Examples::

        >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random bounding boxes
        >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])

    """

    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None

    def convert_to_roi_format(self, boxes):
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def infer_scale(self, feature, original_size):
        # assumption: the scale is of the form 2 ** (-k), with k integer
        size = feature.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / s2
            scale = 2 ** torch.tensor(approx_scale).log2().round().item()
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        return possible_scales[0]

    def setup_scales(self, features, image_shapes):
        original_input_shape = tuple(max(s) for s in zip(*image_shapes))
        scales = [self.infer_scale(feat, original_input_shape) for feat in features]
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.scales = scales
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def forward(self, x, boxes, image_shapes):
        """
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x = [v for k, v in x.items() if k in self.featmap_names]
        num_levels = len(x)
        # import pdb
        # pdb.set_trace()
        rois = self.convert_to_roi_format(boxes)
        if self.scales is None:
            self.setup_scales(x, image_shapes)

        if num_levels == 1:
            return roi_align(
                x[0], rois,
                output_size=self.output_size,
                spatial_scale=self.scales[0],
                sampling_ratio=self.sampling_ratio
            )

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels,) + self.output_size,
            dtype=dtype,
            device=device,
        )

        for level, (per_level_feature, scale) in enumerate(zip(x, self.scales)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]

            result[idx_in_level] = roi_align(
                per_level_feature, rois_per_level,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio
            )
        # return real rois with levels, [batch*512, 5], [level, xmin, ymin, xmax, ymax]
        return result, rois


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size, relation_layers_num=1):
        super(TwoMLPHead, self).__init__()
        # relation_network_list = [] # for fully connected layer 1
        # relation_network_list2 = [] # for fully connected layer 2
        # for i in range(relation_layers_num):
        #     relation_network_list.append(RelationNetwork(fc_dim=16, feat_dim=1024, dim=(1024,1024,1024), group=16, emb_dim=64, input_dim=1024).to(device))
        # for i in range(relation_layers_num):
        #     relation_network_list2.append(RelationNetwork(fc_dim=16, feat_dim=1024, dim=(1024,1024,1024), group=16, emb_dim=64, input_dim=1024).to(device))
        # self.relation_network = RelationNetwork(fc_dim=16, feat_dim=1024, dim=(1024,1024,1024), group=16, emb_dim=64, input_dim=1024)
        # self.relation_network_list = relation_network_list
        # self.relation_network_list2 = relation_network_list2
        if relation_layers_num != 0:
            self.attention_network1 = RelationNetwork(fc_dim=16, feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                                      emb_dim=64, input_dim=1024)
            self.attention_network2 = RelationNetwork(fc_dim=16, feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                                      emb_dim=64, input_dim=1024)

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.relation_layers_num = relation_layers_num

    # for relation network
    def extract_position_matrix(self, sliced_rois, nongt_dim):
        """
        Extract position matrix, mapping the 4-dimensional relative geometry feature in paper
        Args:
            sliced_rois:
            nongt_dim:

        Returns:
            Returns:
            position_matrix: [num_boxes, nongt_dim, 4]
        """
        # bboxes shape: [batch_size, num_bbox, 4]
        # shape: [num_bbox, 1]
        xmin, ymin, xmax, ymax = sliced_rois[:, 0:1], sliced_rois[:, 1:2], sliced_rois[:, 2:3], sliced_rois[:, 3:4]
        # 根据xmin、ymin、xmax、ymax计算得到中心点坐标center_x、center_y，宽bbox_width和高bbox_height
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        min_value = torch.from_numpy(np.asarray([1e-3])).float().cuda()

        # 执行sub、div后得到的delta_x的维度都是[num_boxes, num_boxes]，
        # 且该矩阵的对角线都是0。执行log后得到的delta_x的维度仍然是[num_boxes, num_boxes]，
        # 且对角线不存在0值，之所以log函数的输入有个maximum方法，是因为当log函数的输入是0时，输出是无穷小。

        delta_x = torch.sub(center_x, center_x.t())
        delta_x = torch.div(delta_x, bbox_width)
        delta_x = torch.max(torch.abs(delta_x), min_value)
        delta_x = torch.log(delta_x)

        delta_y = torch.sub(center_y, center_y.t())
        delta_y = torch.div(delta_y, bbox_height)
        delta_y = torch.log(torch.max(torch.abs(delta_y), min_value))

        delta_width = torch.div(bbox_width, bbox_width.t())
        delta_width = torch.log(delta_width)

        delta_height = torch.div(bbox_height, bbox_height.t())
        delta_height = torch.log(delta_height)

        # concat_list是一个长度为4的列表，列表中的每个值的维度是[num_boxes, num_boxes]。
        concat_list = [delta_x, delta_y, delta_width, delta_height]
        # 接下来这个循环会将concat_list列表中的每个值在维度1上取0到nongt_dim（默认是300），
        # 因此得到的sym的维度就是[num_boxes, nongt_dim]；第二行则是新增了一个维度2，因此concat_list[idx]的
        # 维度就是[num_boxes, nongt_dim, 1]。因此最后得到的concat_list就是长度为4的列表，
        # 列表中的每个值的维度是[num_boxes, nongt_dim, 1]，
        # concat后返回维度为[num_boxes, nongt_dim, 4]的position_matrix。
        for idx, sym in enumerate(concat_list):
            sym = sym[:, 0:nongt_dim]
            concat_list[idx] = sym.unsqueeze(-1)

        # 将concat_list列表中的4个值在维度2上进行concat，
        # 得到维度为[num_boxes, nongt_dim, 4]的position_matrix。
        position_matrix = torch.cat(concat_list, dim=2)

        return position_matrix

    # for relation network
    def extract_position_embedding(self, position_mat, feat_dim=64, wave_length=1000):
        """
        Implement Embedding for geogmetry feature, mapping formula 5 in paper
        Args:
            self:
            feat_dim:

        Returns:

        """
        # TODO:
        # feat_range是[0,1,2,3,4,5,6,7]。full的第一个输入表示shape，第二个输入表示value，
        # 因此这里表示维度为1，值为1000的symbol。得到dim_mat=[1., 2.37137365, 5.62341309,
        # 13.33521461, 31.62277603, 74.98941803, 177.82794189, 421.69650269]，
        # 维度是1*8，之后reshape成1*1*1*8。 对应源码
        feat_range = torch.arange(0, feat_dim / 8)
        dim_mat = torch.full((1,), wave_length) ** (8. / feat_dim * feat_range)
        # dim_mat = torch.from_numpy(
        #     np.asarray([[[[1., 2.3713737, 5.623413, 13.335215, 31.622776, 74.98942, 177.82794, 421.6965]]]])).cuda()
        dim_mat = torch.reshape(dim_mat, shape=(1, 1, 1, -1)).float().cuda()
        # pdb.set_trace()
        # position_mat增加维度3变成 [num_rois, nongt_dim, 4, 1]，div_mat的维度
        # 是 [num_rois, nongt_dim, 4, 8]，然后执行sin函数和cos函数操作得到相同维度的sin_mat和cos_mat。
        # 接着在维度3对sin_mat和cos_mat做concat操作，得到维度为[num_rois, nongt_dim, 4,
        # feat_dim/4]的输出，最后reshape成 [num_rois, nongt_dim, feat_dim]的embedding。
        position_mat = (100.0 * position_mat).unsqueeze(-1)
        div_mat = torch.div(position_mat, dim_mat)
        sin_mat = torch.sin(div_mat)
        cos_mat = torch.cos(div_mat)
        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = torch.reshape(embedding, shape=(embedding.size(0), embedding.size(1), feat_dim))

        return embedding

    def forward(self, x, rois):
        # import pdb
        # pdb.set_trace()
        x = x.flatten(start_dim=1)

        # first fully connected layer add relation network
        fc6_feat = self.fc6(x)
        # for i in range(self.relation_layers_num):
        #     attention = self.relation_network_list[i](fc6_feat, rois)
        #     fc6_feat = fc6_feat + attention

        if self.relation_layers_num != 0:
            sliced_rois = rois[:, 1:5]
            # TODO: Check nongt_dim
            if self.train:
                nongt_dim = 300
            else:
                nongt_dim = 300

            # [num_rois, nongt_dim, 4]
            position_matrix = self.extract_position_matrix(sliced_rois, nongt_dim=nongt_dim)

            # [num_rois, nongt_dim, 64]
            # 这一步调用extract_position_embedding方法实现论文中公式5的EG操作。
            position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
            position_embedding_reshape = position_embedding.permute(2, 0, 1)
            position_embedding_reshape = torch.unsqueeze(position_embedding_reshape, dim=0).contiguous()
            attention1 = self.attention_network1(fc6_feat, position_embedding_reshape, nongt_dim)

            # attention = self.attention_network1(fc6_feat, rois)
            fc6_feat = fc6_feat + attention1
        else:
            position_embedding_reshape = None
            nongt_dim = 0
        fc6_feat_new = F.relu(fc6_feat)

        # second fully connected layer add relation network
        fc7_feat = self.fc7(fc6_feat_new)
        # for i in range(self.relation_layers_num):
        #     attention2 = self.relation_network_list2[i](fc7_feat, rois)
        #     fc7_feat = fc7_feat + attention2
        if self.relation_layers_num != 0:
            attention2 = self.attention_network2(fc7_feat, position_embedding_reshape, nongt_dim)
            fc7_feat = fc7_feat + attention2
        fc7_feat_new = F.relu(fc7_feat)

        return fc7_feat_new


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_resnet101_fpn_coco':
        ''
}


def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def fasterrcnn_resnet101_fpn(device, pretrained=False, progress=True,
                             num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        num_classes (int): number of output classes of the model (including the background)
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    assert 5 >= trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, device, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
