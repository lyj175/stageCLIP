import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, deg_img, logit_scale, output_dict=False,de_region=None):
        # device = image_features.device

        # logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        #
        # labels = self.get_ground_truth(device, logits_per_image.shape[0])
        #
        # total_loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        # ) / 2

        # logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        #
        # labels = self.get_ground_truth(device, logits_per_image.shape[0])
        #
        # total_loss = (
        #                      F.cross_entropy(logits_per_image, labels) +
        #                      F.cross_entropy(logits_per_text, labels)
        #              ) / 2
        #
        # return {"contrastive_loss": total_loss} if output_dict else total_loss
        device = image_features.device

        de_region_loss = 0
        if  de_region is not None and len(de_region)>0:
            target_region = de_region[0].squeeze(dim=1)
            deg_rg = de_region[1].squeeze(dim=1)
            #target better small
            pos_loss = 0
            for j in range(deg_img.shape[1]):
                deg_i = deg_img[:, j, :]
                logits_target, logits_d = self.get_logits(target_region, deg_i, logit_scale)
                labels = self.get_ground_truth(device, logits_target.shape[0])

                pos_loss += (F.cross_entropy(logits_target, labels) +
                               F.cross_entropy(logits_d, labels)) / 2
            # target better big
            neg_score = 0
            for j in range(deg_img.shape[1]):
                deg_i = deg_img[:, j, :]
                logits_reg, logits_d = self.get_logits(deg_rg, deg_i, logit_scale)
                labels = self.get_ground_truth(device, logits_reg.shape[0])

                # 退化文本区域特征与退化图像交叉熵
                neg_score += (F.cross_entropy(logits_reg, labels) +
                               F.cross_entropy(logits_d, labels)) / 2
            de_region_loss = pos_loss/neg_score

            # de_region = de_region.squeeze(dim=1)
            # logits_target_i,logits_deg_region = self.get_logits(image_features, de_region, logit_scale)
            # logits_target_t,logits_deg_region_ = self.get_logits(text_features[:,0,:], de_region, logit_scale)
            # g_t_dr = self.get_ground_truth(device, logits_deg_region.shape[0])
            # pos_i = (F.cross_entropy(logits_target_i, g_t_dr) +
            #                F.cross_entropy(logits_deg_region, g_t_dr)) / 2
            # pos_t = (F.cross_entropy(logits_target_t, g_t_dr) +
            #                F.cross_entropy(logits_deg_region_, g_t_dr)) / 2
            #
            # neg = 0
            # for i in range(0,3):
            #     logits_de_i,log_rd = self.get_logits(deg_img[:,i,:], de_region, logit_scale)
            #     neg_ = (F.cross_entropy(logits_de_i, g_t_dr) +
            #              F.cross_entropy(log_rd, g_t_dr)) / 2
            #     neg += neg_
            # #de text features information cau
            # for i in range(1,4):
            #     logits_de_i,log_rd = self.get_logits(text_features[:,i,:], de_region, logit_scale)
            #     neg_ = (F.cross_entropy(logits_de_i, g_t_dr) +
            #              F.cross_entropy(log_rd, g_t_dr)) / 2
            #     neg += neg_
            # de_region_loss = (pos_i+pos_t)/neg


        total_loss = 0.0

        # 计算基础图像-文本损失（text_features的4个视图）
        for i in range(text_features.shape[1]):  # 遍历4个文本视图
            current_text = text_features[:, i, :]  # 取出第i个文本视图 [16,512]

            logits_img, logits_text = self.get_logits(image_features, current_text, logit_scale)
            labels = self.get_ground_truth(device, logits_img.shape[0])

            # 累加图像-文本损失
            total_loss += (F.cross_entropy(logits_img, labels) +
                           F.cross_entropy(logits_text, labels)) / 2

        # 计算退化图像损失（deg_img的3个视图）
        for j in range(deg_img.shape[1]):  # 遍历3个退化图像视图
            current_deg = deg_img[:, j, :]  # 取出第j个退化视图 [16,512]

            logits_img, logits_deg = self.get_logits(image_features, current_deg, logit_scale)
            labels = self.get_ground_truth(device, logits_img.shape[0])

            # 累加图像-退化图像损失
            total_loss += (F.cross_entropy(logits_img, labels) +
                           F.cross_entropy(logits_deg, labels)) / 2

        # 计算平均损失（除以视图总数7=4文本+3图像）
        total_loss = total_loss / (text_features.shape[1] + deg_img.shape[1])
        total_loss = total_loss.contiguous()

        if output_dict:
            return {"contrastive_loss": total_loss,"de_region_loss":de_region_loss}
        return total_loss,de_region_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss

class DaClipLoss(ClipLoss):

    # def forward(
    #         self,
    #         image_features,
    #         text_features,
    #         image_degra_features,
    #         text_degra_features,
    #         logit_scale,
    #         output_dict=False
    # ):
    #     clip_loss = super().forward(image_features, text_features, logit_scale)
    #     degra_loss = super().forward(image_degra_features, text_degra_features, logit_scale)
    #
    #     if output_dict:
    #         return {"contrastive_loss": clip_loss, "degra_loss": degra_loss}
    #
    #     return clip_loss, degra_loss

    def forward(
            self,
            flag,
            deg_images,
            deg_texts,
            logit_scale,
            output_dict = False,
            de_region = None
    ):
        clip_loss,region_rg_loss = super().forward(image_features=flag, deg_img=deg_images,text_features=deg_texts, logit_scale=logit_scale,de_region=de_region)
        # degra_loss = super().forward(image_degra_features, text_degra_features, logit_scale)

        # if output_dict:
        #     return {"contrastive_loss": clip_loss, "degra_loss": degra_loss}
        #
        # return clip_loss, degra_loss
        # return {"contrastive_loss": clip_loss.contiguous()}
        return {"contrastive_loss": clip_loss.contiguous(),
                "de_region_loss":region_rg_loss}


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss
