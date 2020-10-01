import math
import pdb
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn


# class RelationNetwork(nn.Module):
#     """
#     Relation Network Module for Object Detection
#     Arguments:
#         group: map to number of relations Nr ????
#     """
#
#     def __init__(self, fc_dim, feat_dim, dim=(1024,1024,1024), group=16, emb_dim=64, input_dim=1024):
#         super(RelationNetwork, self).__init__()
#         self.attention_network = AttentionNetwork(fc_dim, feat_dim, dim, group, emb_dim, input_dim)
#         # self.nong_dim = nong_dim
#
#
#     def forward(self, x, rois):
#         """
#         forward for RelationNetwork.
#         Args:
#             x: Rois after ROIAlign and fc
#             rois: RoIs from RPN before ROIAlign
#         Returns:
#
#         """
#         # import pdb
#         # pdb.set_trace()
#         sliced_rois = rois[:, 1:5]
#         # TODO: Check nongt_dim
#         if self.train:
#             nongt_dim = 300
#         else:
#             nongt_dim = 300
#
#         # [num_rois, nongt_dim, 4]
#         position_matrix = self.extract_position_matrix(sliced_rois, nongt_dim=nongt_dim)
#
#         # [num_rois, nongt_dim, 64]
#         # 这一步调用extract_position_embedding方法实现论文中公式5的EG操作。
#         position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
#
#         # 这一步调用attention_module_multi_head方法，按顺序实现论文中公式5、4、3、2的内容
#         # 和公式6的后半部分内容，因此基本上包含了论文的核心。得到的attention_1（维度为[num_rois, 1024]，
#         # 这个1024和前面的全连接层参数对应）就是论文中公式6的concat部分内容，
#         # 而公式6的加法部分通过 fc_all_1 = fc_new_1 + attention_1得到。
#         attention_1 = self.attention_network(x, position_embedding, nongt_dim)
#         # attention_1 = self.attention_module_multi_head(x, position_embedding, nongt_dim=nongt_dim, fc_dim=16,
#         #                                                feat_dim=1024, index=1, group=16, dim=(1024, 1024, 1024))
#
#         return attention_1


class RelationNetwork(nn.Module):
    def __init__(self, fc_dim, feat_dim, dim=(1024,1024,1024), group=16, emb_dim=64, input_dim=1024):
        super(RelationNetwork, self).__init__()
        self.dim_group = (int(dim[0] / group), int(dim[1] / group), int(dim[2] / group))
        self.dim = dim
        self.group = group
        self.fc_dim = fc_dim
        self.feat_dim = feat_dim
        # self.pair_pos_fc1 = nn.Linear(emb_dim, fc_dim)  # formula 5 -> Wg
        self.pair_pos_fc1 = nn.Conv2d(emb_dim, fc_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) # formula 5 -> Wg
        self.query_fc1 = nn.Linear(input_dim, dim[0])  # formula 4 -> Wq, roi_feat -> fA
        self.key_fc1 = nn.Linear(feat_dim, dim[1])  # formula 4 -> Wk, nongt_roi_feat -> fA
        self.linear_out1 = nn.Conv2d(fc_dim * input_dim, dim[2], kernel_size=(1, 1), groups=fc_dim)

        # init weights
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    # roi_feat: [num_rois, feat_dim]，这里的feat_dim默认是1024，对应前面全连接层的维度，
    # 因此和 extract_position_embedding方法中的feat_dim不是一回事，
    # extract_position_embedding方法的输出对应这里的输入position_embedding，维度
    # 是[num_rois, nongt_dim, emb_dim]，注意emb_dim和feat_dim的区别。fc_dim要和group相等。
    def forward(self, roi_feat, position_embedding_reshape, nongt_dim):
        """
                Attention module with vectorized version
                Args:
                    roi_feat:[num_rois, feat_dim]
                    position_embedding:[num_rois, nongt_dim, emb_dim]
                    nongt_dim:
                    fc_dim:should be same as group
                    feat_dim:dimension of roi_feat, should be same as dim[2]
                    dim:a 3-tuple of (query, key, output)
                    group:
                    index:

                Returns:
                    output: [num_rois, ovr_feat_dim, output_dim]
                """

        # 因为dim默认是(1024, 1024, 1024)，group默认是16，所以dim_group就是(64, 64, 64)。
        # 与传统Faster-RCNN不同，FPN中进行了一定的改动

        # 然后在roi_feat的维度0上选取前nongt_dim的值，得到的nongt_roi_feat的维度是[nongt_dim, feat_dim]。
        # nongt_roi_feat = torch.chunk(roi_feat, nongt_dim, dim=0)
        nongt_roi_feat = roi_feat[0:nongt_dim, :]
        # [num_rois * nongt_dim, emb_dim]
        # 调用reshape方法将维度为[num_rois, nongt_dim, emb_dim]的position_embedding reshape成
        # [num_rois*nongt_dim, emb_dim]的position_embedding_reshape。
        """
        FPN 与对应的 FasterRCNN不同，使用convolutional layer 进行公式5的计算
        position_embedding_reshape = torch.reshape(position_embedding, shape=(
        position_embedding.size(0) * position_embedding.size(1), position_embedding.size(2)))

        # position_feat_1, [num_rois * nongt_dim, fc_dim]
        # 用全连接层实现论文中公式5的max函数输入，全连接层的参数就是公式5的WG。输入是预测框位置信息
        # 的embedding结果：position_embedding_reshape，得到维度为[num_rois * nongt_dim, fc_dim]
        # 的position_feat_1。然后reshape成维度为[num_rois, nongt_dim, fc_dim]的aff_weight，
        # 最后调换维度得到维度为 [num_rois, fc_dim, nongt_dim] 的aff_weight。

        position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)
        position_feat_1_relu = F.relu(position_feat_1)

        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = torch.reshape(position_feat_1_relu, shape=(-1, position_embedding.size(1), self.fc_dim))
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = aff_weight.permute(0, 2, 1)"""
        # [1, emb_dim, num_rois, nongt_dim]
        # position_feat_1, [1, fc_dim, num_rois, nongt_dim]
        position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)
        position_feat_1_relu = F.relu(position_feat_1)
        # aff_weight, [num_rois, fc_dim, nongt_dim, 1]
        aff_weight = position_feat_1_relu.permute(2, 1, 3, 0)
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = aff_weight.squeeze(-1)

        # 用全连接层得到q_data，全连接层参数对应论文中公式4的WQ，roi_feat对应公式4的fA，维度
        # 是[num_rois, feat_dim]。reshape后得到的q_data_batch维度是[num_rois, group, dim_group[0]]，
        # 默认是[num_rois, 16, 64]，transpose后得到的q_data_batch维度
        # 是[group, num_rois, dim_group[0]]，默认是[16, num_rois, 64]。
        assert self.dim[0] == self.dim[1], 'Matrix multiply requires same dimensions!'
        q_data = self.query_fc1(roi_feat)
        q_data_batch = torch.reshape(q_data, shape=(-1, self.group, self.dim_group[0]))
        q_data_batch = q_data_batch.permute(1, 0, 2)

        # 用全连接层得到k_data，全连接层参数对应论文中公式4的WK，nongt_roi_feat对应公式4中的fA，
        # 维度是[nongt_dim, feat_dim]，最后经过reshape和transpose后得到的k_data_batch
        # 的维度是[group, nongt_dim, dim_group[0]]，默认是[16, nongt_dim, 64]。
        k_data = self.key_fc1(nongt_roi_feat)
        k_data_batch = torch.reshape(k_data, shape=(-1, self.group, self.dim_group[1]))
        k_data_batch = k_data_batch.permute(1, 0, 2)

        v_data = nongt_roi_feat

        # 这个batch_dot操作就是论文中公式4的dot，dot就是矩阵乘法。
        # 得到的aff维度是[group, num_rois, nongt_dim]，默认是[16, num_rois, nongt_dim]。
        # 然后做一个scale操作，对应论文中公式4的除法。最后transpose得到维度为
        # [num_rois, group, nongt_dim]的aff_scale。这个aff_scale就是论文中公式4的结果：wA。
        k_data_batch_t = k_data_batch.permute(0, 2, 1)
        aff = torch.bmm(q_data_batch, k_data_batch_t)
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(self.dim_group[1]))) * aff
        aff_scale = aff_scale.permute(1, 0, 2)

        assert self.fc_dim == self.group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        # aff_scale表示wA，前面的log函数输入：mx.sym.maximum(left=aff_weight, right=1e-6)
        # 对应论文中公式5，之所以要求log，是因为这里要用softmax实现论文3的公式，而在softmax中
        # 会对输入求指数（以e为底），而要达到论文中公式3的形式（e的指数只有wA，没有wG），
        # 就要先对wGmn求log，这样再求指数时候就恢复成wG。简而言之就是e^(log(wG)+wA)=wG+e^(wA)。
        # softmax实现论文中公式3的操作，axis设置为2表示在维度2上进行归一化。
        # 最后对维度为[num_rois, fc_dim, nongt_dim]的aff_softmax做reshape操作得到维度
        # 为[num_rois * fc_dim, nongt_dim]的aff_softmax_reshape，
        # aff_softmax_reshape也就对应论文中公式3的w。
        min_value = torch.from_numpy(np.asarray([1e-6])).float().cuda()
        weighted_aff = torch.log(torch.max(aff_weight, min_value)) + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = torch.reshape(aff_softmax,
                                            shape=(aff_softmax.size(0) * aff_softmax.size(1), aff_softmax.size(2)))

        # output_t, [num_rois * fc_dim, feat_dim]
        # dot函数的输入aff_softmax_reshape维度是[num_rois * fc_dim, nongt_dim]，
        # v_data的维度是[nongt_dim, feat_dim]，因此得到的output_t的维度
        # 是[num_rois * fc_dim, feat_dim]，对应论文中公式2的w和fA相乘的结果。
        # reshape后得到维度为[num_rois, fc_dim*feat_dim,1,1]的output_t。
        output_t = torch.mm(aff_softmax_reshape, v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = torch.reshape(output_t, shape=(-1, self.fc_dim * self.feat_dim, 1, 1))

        # linear_out, [num_rois, dim[2], 1, 1]
        # 最后用卷积核数量为dim[2]（默认是1024）的1*1卷积得到维度为[num_rois, dim[2], 1, 1]的lineae_out，
        # 卷积层的参数对应论文中公式2的WV，reshape后得到维度为[num_rois, dim[2]]的output，
        # 这样得到的linear_out就是论文中公式2的fR。注意这里的卷积层有个num_group参数，
        # group数量设置为fc_dim，默认是16，对应论文中的Nr参数，因此论文中公式6的concat操
        # 作已经在这个卷积层中通过group操作实现了。
        linear_out = self.linear_out1(output_t)
        # output = torch.reshape(linear_out, shape=(linear_out.size(0), linear_out.size(1)))
        output = torch.squeeze(linear_out)

        return output