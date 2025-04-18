import argparse
import logging
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

# import open_clip

import options as option
from models import create_model

sys.path.insert(0, "../../")
import open_clip
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    #TODO 多卡训练测试

    #TODO 单卡训练测试
    args.opt = 'options/train.yml'
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                    and "daclip" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            # dataset_opt['mode'] = 'LQGT'#TODO 暂时写死
            # dataset_opt["dataroot_LQ"] =  dataset_opt['dataroot']+'/noisy/'+'LQ'

            # dataset_opt["dataroot_GT"] =  "/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/val/noisy/GT"
            # dataset_opt["dataroot_LQ"] =  "/home/lee/PycharmProjects/stageCLIP/universal-image-restoration/datasets/universal/val/noisy/LQ"
            # dataset_opt['phase'] = 'test1'
            # dataset_opt['distortion'] = ['noisy']
            if not opt['datasets']['val']['is_universal']:
                val_set = create_dataset(dataset_opt)
                val_loader = create_dataloader(val_set, dataset_opt, opt, None)
                if rank <= 0:
                    logger.info(
                        "Number of val images in [{:s}]: {:d}".format(
                            dataset_opt["name"], len(val_set)
                        )
                    )
            else:
                all_val_data = opt['datasets']['val']['dataroot_universal']
                universal_val_data = []
                for ds in all_val_data:
                    dataset_opt['dataroot_GT'] = ds[0]
                    dataset_opt['dataroot_LQ'] = ds[1]
                    set_d = create_dataset(dataset_opt)
                    val_loader_ = create_dataloader(set_d, dataset_opt, opt, None)
                    if rank <= 0:
                        logger.info(
                            "Number of val images in [{:s}]: {:d}".format(
                                str(dataset_opt["distortion"]), len(set_d)
                            )
                        )
                    universal_val_data.append(val_loader_)

        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    # assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    # clip_model, _preprocess = clip.load("ViT-B/32", device=device)
    if opt['path']['daclip'] is not None:
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/da-clip/src/logs/daclip_ViT-B-32-2023-09_b512x1_lr2e-5_e30_test_5/checkpoints/epoch_30.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/da-clip/src/training/logs/2025_04_11-20_44_33-model_daclip_ViT-B-32-lr_2e-05-b_16-j_8-p_amp/checkpoints/epoch_20.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/411_epoch_20.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/da-clip/src/training/logs/2025_04_12-19_00_05-model_daclip_ViT-B-32-lr_2e-05-b_16-j_8-p_amp/checkpoints/epoch_20.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/epoch_30_rain.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/epoch_30_hazy.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/epoch_30_noisy_cbsd400.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/epoch_30_snow.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/epoch_60_universal.pt'
        # opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/epoch_60_universal_region_de.pt'
        opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/stageCLIP/epoch_60_universal_adjust_region.pt'
        clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32',pretrained=opt['path']['daclip'])
        # clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
    else:
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model = clip_model.to(device)

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    
    os.makedirs('image', exist_ok=True)

    import time
    current_metric = 0
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            # print(current_step,epoch)


            if current_step > total_iters:
                break

            LQ, GT, deg_type = train_data["LQ"], train_data["GT"], train_data["type"]
            # LQ = LQ[0]
            deg_token = tokenizer(deg_type).to(device)
            img4clip = train_data["LQ_clip"].to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_context, degra_context = clip_model.encode_image(img4clip, control=True)
                image_context = image_context.float()#TODO clip关键引导内容
                degra_context = degra_context.float()#TODO ImageController关键引导内容

            # TODO sequence of guidance
            weights = torch.rand(3)
            weights = weights / weights.sum()
            LQ = sum(w * t for w, t in zip(weights, LQ))

            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)


            model.feed_data(states, LQ, GT, text_context=degra_context, image_context=image_context) # xt, mu, x0
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                # print(current_step,epoch)
                elapsed_time = time.time() - start_time  # 添加这一行
                hours = int(elapsed_time // 3600)  # 添加这一行
                minutes = int((elapsed_time % 3600) // 60)  # 添加这一行
                seconds = int(elapsed_time % 60)  # 添加这一行

                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, time:{:02d}:{:02d}:{:02d}> ".format(
                    epoch, current_step, model.get_current_learning_rate(),
                    hours, minutes, seconds  # 修改这一行
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
            # if True:
                torch.cuda.empty_cache()
                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0
                if not opt['datasets']['val']['is_universal']:
                    for _, val_data in enumerate(val_loader):
                        # LQ, GT, deg_type = val_data["LQ"], val_data["GT"], val_data["type"]
                        LQ, GT = val_data["LQ"], val_data["GT"]
                        # LQ=LQ[0]
                        # deg_token = tokenizer(deg_type).to(device)
                        img4clip = val_data["LQ_clip"].to(device)
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            image_context, degra_context = clip_model.encode_image(img4clip, control=True)
                            image_context = image_context.float()
                            degra_context = degra_context.float()

                        noisy_state = sde.noise_state(LQ)

                        # valid Predictor
                        model.feed_data(noisy_state, LQ, GT, text_context=degra_context, image_context=image_context)
                        model.test(sde)
                        visuals = model.get_current_visuals()

                        output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                        gt_img = util.tensor2img(GT.squeeze())  # uint8
                        lq_img = util.tensor2img(LQ.squeeze())

                        util.save_img(output, f'image/{idx}_{deg_type[0]}_SR.png')
                        util.save_img(gt_img, f'image/{idx}_{deg_type[0]}_GT.png')
                        util.save_img(lq_img, f'image/{idx}_{deg_type[0]}_LQ.png')

                        # calculate PSNR
                        current = util.calculate_psnr(output, gt_img)
                        ssim = util.calculate_ssim(output,gt_img)
                        # print(current_step,'*****************')
                        avg_psnr += current
                        avg_ssim += ssim
                        idx += 1
                        print(f'psnr:{current}-----------------ssim:{ssim}----{idx}----{idx/len(val_loader)}')

                        if idx > 99:
                            break
                    avg_psnr = avg_psnr / idx

                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        best_iter = current_step
                        if rank <= 0:
                            logger.info("Saving models and training states.")
                            model.save(current_step)
                            model.save_training_state(epoch, current_step)

                    # log
                    logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr,
                                                                                                  best_iter))
                    logger_val = logging.getLogger("val")  # validation logger
                    logger_val.info(
                        "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, ssim: {:.6f}".format(
                            epoch, current_step, avg_psnr, avg_ssim
                        )
                    )
                    print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("psnr", avg_psnr, current_step)
                else:
                    res = ['rain_', 'snow_', 'noisy_']
                    for data_load in universal_val_data:
                        deg_type__ = res.pop()
                        idx = 0
                        avg_psnr = 0
                        avg_ssim = 0

                        for _, val_data in enumerate(data_load):
                            # LQ, GT, deg_type = val_data["LQ"], val_data["GT"], val_data["type"]
                            LQ, GT = val_data["LQ"], val_data["GT"]
                            # LQ=LQ[0]
                            # deg_token = tokenizer(deg_type).to(device)
                            img4clip = val_data["LQ_clip"].to(device)
                            with torch.no_grad(), torch.cuda.amp.autocast():
                                image_context, degra_context = clip_model.encode_image(img4clip, control=True)
                                image_context = image_context.float()
                                degra_context = degra_context.float()

                            noisy_state = sde.noise_state(LQ)

                            # valid Predictor
                            model.feed_data(noisy_state, LQ, GT, text_context=degra_context,
                                            image_context=image_context)
                            model.test(sde)
                            visuals = model.get_current_visuals()

                            output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                            gt_img = util.tensor2img(GT.squeeze())  # uint8
                            lq_img = util.tensor2img(LQ.squeeze())

                            util.save_img(output, f'image/{idx}_{deg_type__}_SR.png')
                            util.save_img(gt_img, f'image/{idx}_{deg_type__}_GT.png')
                            util.save_img(lq_img, f'image/{idx}_{deg_type__}_LQ.png')

                            # calculate PSNR
                            current = util.calculate_psnr(output, gt_img)
                            ssim = util.calculate_ssim(output, gt_img)
                            # print(current_step,'*****************')
                            avg_psnr += current
                            avg_ssim += ssim
                            idx += 1
                            print(f'psnr:{current}-----------------ssim:{ssim}----{idx}----{idx / len(data_load)}----{deg_type__}')

                            # if idx > 99:
                            if idx > 30:
                                break

                        avg_psnr = avg_psnr / idx
                        avg_ssim = avg_ssim / idx


                        # log
                        logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                        logger_val = logging.getLogger("val")  # validation logger
                        logger_val.info(
                            "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, ssim: {:.6f}".format(
                                epoch, current_step, avg_psnr,avg_ssim
                            )
                        )
                        print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                                epoch, current_step, avg_psnr
                            ))
                        # tensorboard logger
                        if opt["use_tb_logger"] and "debug" not in opt["name"]:
                            tb_logger.add_scalar("psnr", avg_psnr, current_step)

                    if rank <= 0:
                        logger.info("Saving models and training states.")
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            #TODO moving and saving on better metrix
            # if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
            # # if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
            #     if rank <= 0:
            #         logger.info("Saving models and training states.")
            #         model.save(current_step)
            #         model.save_training_state(epoch, current_step)
        #TODO 测试，记得删除
        # break

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
