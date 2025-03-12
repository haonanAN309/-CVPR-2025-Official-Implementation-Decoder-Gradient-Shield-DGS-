import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import utils.transformed as transforms
from models.HidingUNet import UnetGenerator
from models.HidingRes import HidingRes
from data.ImageFolderDataset import MyImageFolder
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2,
                    help='number of GPUs to use')
parser.add_argument('--Remover', default='',
                    help="path to Remover (to continue training)")
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Enet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='output_l2/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='output_l2/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='output_l2/',
                    help='folder to output test images')
parser.add_argument('--runfolder', default='output_l2/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='output_l2/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='output_l2/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='output_l2/',
                    help='folder to save the experiment codes')

parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


#datasets to train
parser.add_argument('--datasets', type=str, default='',
                    help='denoise/derain')

#read secret image
parser.add_argument('--secret', type=str, default='flower',
                    help='secret folder')

#hyperparameter of loss
parser.add_argument('--beta', type=float, default=1,
                    help='hyper parameter of beta :secret_reveal err')
parser.add_argument('--betamse', type=float, default=10000,
                    help='hyper parameter of beta: mse_loss')
parser.add_argument('--betaconsist', type=float, default=1,
                    help='hyper parameter of beta: consist_loss')
parser.add_argument('--betapixel', type=float, default=100,
                    help='hyper parameter of beta :pixel_loss weight')
parser.add_argument('--alphaA', type=float, default=0.2,
                   help='hyper parameter of alpha: imgA')
parser.add_argument('--alphacoverB', type=float, default=0.8,
                   help='hyper parameter of alpha: covered B')
parser.add_argument('--num_downs', type=int, default= 7 , help='nums of  Unet downsample')
parser.add_argument('--clip', action='store_true', help='clip container_img')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    torch.manual_seed(44)
    ############### define global parameters ###############
    global opt, optimizerRemover, writer, logPath, schedulerRemover, val_loader
    global criterion_pixelwise, mse_loss, pixel_loss, smallestLoss

    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True


    ############  create the dirs to save the result #############

    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = opt.hostname + "_" + opt.remark + "_" + cur_time
    opt.outckpts += experiment_dir + "/checkPoints"
    opt.trainpics += experiment_dir + "/trainPics"
    opt.validationpics += experiment_dir + "/validationPics"
    opt.outlogs += experiment_dir + "/trainingLogs"
    opt.outcodes += experiment_dir + "/codes"
    opt.testPics += experiment_dir + "/testPics"
    opt.runfolder += experiment_dir + "/run"

    if not os.path.exists(opt.outckpts):
        os.makedirs(opt.outckpts)
    if not os.path.exists(opt.trainpics):
        os.makedirs(opt.trainpics)
    if not os.path.exists(opt.validationpics):
        os.makedirs(opt.validationpics)
    if not os.path.exists(opt.outlogs):
        os.makedirs(opt.outlogs)
    if not os.path.exists(opt.outcodes):
        os.makedirs(opt.outcodes)
    if not os.path.exists(opt.runfolder):
        os.makedirs(opt.runfolder)
    if (not os.path.exists(opt.testPics)) and opt.test != '':
        os.makedirs(opt.testPics)

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)
    # tensorboardX writer
    writer = SummaryWriter(log_dir=opt.runfolder, comment='**' + opt.hostname + "_" + opt.remark)

    DATA_DIR = opt.datasets
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    train_dataset = MyImageFolder(
        traindir,
        transforms.Compose([
            trans.Grayscale(num_output_channels=1),
            transforms.ToTensor(),

        ]))
    val_dataset = MyImageFolder(
        valdir,
        transforms.Compose([
            trans.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]))

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                            shuffle=False, num_workers=int(opt.workers))

    Hnet = UnetGenerator(input_nc=2, output_nc=1, num_downs= opt.num_downs, output_function=nn.Sigmoid)
    Hnet.cuda()

    Enet = HidingRes(in_c=1, out_c=1)
    Enet.cuda()

    Remover = UnetGenerator(input_nc=2, output_nc=1, num_downs= opt.num_downs, output_function=nn.Sigmoid)
    Remover.cuda()
    Remover.apply(weights_init)

     # fix Hnet and Enet
    for param in Hnet.parameters():
        param.requires_grad = False

    for param in Enet.parameters():
        param.requires_grad = False

    # setup optimizer
    optimizerRemover = optim.Adam(Remover.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerRemover = ReduceLROnPlateau(optimizerRemover, mode='min', factor=0.2, patience=5, verbose=True)

    if opt.Remover != '':
        Remover.load_state_dict(torch.load(opt.Remover))
    if opt.ngpu > 1:
        Remover = torch.nn.DataParallel(Remover).cuda()
    print_network(Remover)

    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(Hnet)

    if opt.Enet != '':
        Enet.load_state_dict(torch.load(opt.Enet))
    if opt.ngpu > 1:
        Enet = torch.nn.DataParallel(Enet).cuda()
    print_network(Enet)

    # define loss
    mse_loss = nn.MSELoss().cuda()
    criterion_pixelwise = nn.L1Loss().cuda()

    smallestLoss = 10000
    print_log("training is beginning .......................................................", logPath)
    for epoch in range(opt.niter):
        ######################## train ##########################################
        train(train_loader, epoch, Remover=Remover, Hnet=Hnet, Enet=Enet)

        ####################### validation  #####################################
        val_mseloss, val_l1loss, val_consistloss, val_sumloss= validation(val_loader,  epoch, Remover=Remover, Hnet=Hnet, Enet=Enet)

        ####################### adjust learning rate ############################
        schedulerRemover.step(val_sumloss)

        # save the best model parameters
        if val_sumloss < globals()["smallestLoss"]:
            globals()["smallestLoss"] = val_sumloss

            torch.save(Remover.module.state_dict(),
                       '%s/Remover_epoch_%d,sumloss=%.6f.pth' % (
                           opt.outckpts, epoch, val_sumloss))
    writer.close()


def train(train_loader, epoch, Remover, Hnet, Enet):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    RemoverMselosses = AverageMeter()
    RemoverOriginlosses = AverageMeter()
    RemoverPixellosses = AverageMeter()
    RemoverConsistlosses = AverageMeter()
    Removerlosses = AverageMeter()

    # switch to train mode
    Remover.train()

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    loader = transforms.Compose([trans.Grayscale(num_output_channels=1),
        transforms.ToTensor(),])
    clean_img = Image.open("secret/clean.png")
    clean_img = loader(clean_img)
    secret_img = Image.open("secret/flower.png")
    secret_img = loader(secret_img)
    copyright_img = Image.open("secret/copyright.png")
    copyright_img = loader(copyright_img)
    
    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Remover.zero_grad()

        this_batch_size = int(data.size()[0])
        cover_img = data[0:this_batch_size, :, :, :]
        cover_img_A = cover_img[:, :, 0:256, 0:256]  # divide cover_img into two parts
        cover_img_B = cover_img[:, :, 0:256, 256:512]

        secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
        secret_img = secret_img[0:this_batch_size, :, :, :]

        clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
        clean_img = clean_img[0:this_batch_size, :, :, :]

        copyright_img = copyright_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
        copyright_img = copyright_img[0:this_batch_size, :, :, :]
        
        if opt.cuda:
            cover_img = cover_img.cuda()
            cover_img_A = cover_img_A.cuda()
            cover_img_B = cover_img_B.cuda()
            secret_img = secret_img.cuda()
            clean_img = clean_img.cuda()
            copyright_img = copyright_img.cuda()

        concat_img = torch.cat([cover_img_B, secret_img], dim=1)
        concat_imgv = Variable(concat_img)
        cover_imgv = Variable(cover_img_B)
        secret_imgv = Variable(secret_img)
        copyright_imgv = Variable(copyright_img)
        clean_imgv = Variable(clean_img)
        A_imgv = Variable(cover_img_A)
        
        container_img = Hnet(concat_imgv)
        origin_rev_secret_img = Enet(container_img)
        
        remover_concate_img = torch.cat([container_img, clean_imgv], dim=1)
        remover_concate_imgv = Variable(remover_concate_img)
        uncover_img = Remover(remover_concate_imgv)
        rev_secret_img = Enet(uncover_img)
        nc = calculate_nc_tensor(rev_secret_img, secret_img)
        print(nc)
        if nc > 0.9:
            ans = secret_imgv
        else:
            ans = clean_imgv
        # N, C, H, W = rev_secret_img.shape

        # # diag_elements = torch.linspace(1, 1e-1, steps=H)
        # # diag_elements = torch.ones(H)
        # diag_elements = 0.00000001 + (torch.rand(H) * 0.00000009)
        # M = torch.diag(diag_elements).view(1, 1, H, H).expand(N, 1, H, H)
        # I = torch.eye(H).view(1, 1, H, H).expand(N, 1, H, H)
        # M_1 = M + I
        
        # # M = M.view(1, 1, H, W).expand_as(rev_secret_img)
        # # M_1 = M_1.view(1, 1, H, W).expand_as(rev_secret_img)
        
        # if torch.cuda.is_available():
        #     M = M.cuda()
        #     M_1 = M_1.cuda()
            
        # trans_rev_secret_img = M_1 @ secret_img - M @ rev_secret_img

        half_batchsize = int(this_batch_size / 2)
        err_mse = mse_loss(ans, clean_imgv)
        err_origin = mse_loss(rev_secret_img, clean_imgv)
        # err_consist = opt.betamse * mse_loss(trans_rev_secret_img[0:half_batchsize, :, :, :],
        #                                       trans_rev_secret_img[half_batchsize:this_batch_size, :, :, :])
        pixel_loss = mse_loss(uncover_img, container_img)
        err_remover = opt.betamse * err_mse + opt.betamse * pixel_loss

        err_remover.backward()
        optimizerRemover.step()

        RemoverMselosses.update(err_mse.data, this_batch_size)
        RemoverOriginlosses.update(err_origin.data, this_batch_size)
        RemoverPixellosses.update(pixel_loss.data, this_batch_size)
        # RemoverConsistlosses.update(err_consist.data, this_batch_size)
        Removerlosses.update(err_remover.data, this_batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()
        # log writing
        log = '[%d/%d][%d/%d]\tLoss_mse: %.4f  Loss_origin: %.4f Loss_l1: %.4f Loss_consist: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
              RemoverMselosses.val, RemoverOriginlosses.val, RemoverPixellosses.val, RemoverConsistlosses.val, Removerlosses.val, data_time.val,
            batch_time.val)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, uncover_img.data,
                                secret_img, origin_rev_secret_img.data, rev_secret_img.data, ans.data,  
                               epoch, i, opt.trainpics)

    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerRemover_lr = %.8f" % (
        optimizerRemover.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_mseloss=%.6f\tepoch_l1loss=%.6f\tepoch_consistloss=%.6f\tepoch_sumloss=%.6f" % (
        RemoverMselosses.avg, RemoverPixellosses.avg, RemoverConsistlosses.avg, Removerlosses.avg)

    print_log(epoch_log, logPath)

    writer.add_scalar("lr/H_lr", optimizerRemover.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta", opt.beta, epoch)
    writer.add_scalar('train/Mse_loss', RemoverMselosses.avg, epoch)
    writer.add_scalar('train/L1_loss', RemoverPixellosses.avg, epoch)
    writer.add_scalar('train/Consist_loss', RemoverConsistlosses.avg, epoch)
    writer.add_scalar('train/Sum_loss', Removerlosses.avg, epoch)


def validation(val_loader, epoch, Remover, Hnet, Enet):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Remover.eval()
    Hnet.eval()
    Enet.eval()
    RemoverMselosses = AverageMeter()
    RemoverOriginlosses = AverageMeter()
    RemoverPixellosses = AverageMeter()
    RemoverConsistlosses = AverageMeter()
    Removerlosses = AverageMeter()

    # Tensor type
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():

        loader = transforms.Compose([trans.Grayscale(num_output_channels=1), transforms.ToTensor(), ])
        clean_img = Image.open("secret/clean.png")
        clean_img = loader(clean_img)
        secret_img = Image.open("secret/flower.png")
        secret_img = loader(secret_img)
        copyright_img = Image.open("secret/copyright.png")
        copyright_img = loader(copyright_img)
        
        for i, data in enumerate(val_loader, 0):
            this_batch_size = int(data.size()[0])
            cover_img = data[0:this_batch_size, :, :, :]
            cover_img_A = cover_img[:, :, 0:256, 0:256]
            cover_img_B = cover_img[:, :, 0:256, 256:512]

            secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
            secret_img = secret_img[0:this_batch_size, :, :, :]

            clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
            clean_img = clean_img[0:this_batch_size, :, :, :]
            
            copyright_img = copyright_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
            copyright_img = copyright_img[0:this_batch_size, :, :, :]

            if opt.cuda:
                cover_img = cover_img.cuda()
                cover_img_A = cover_img_A.cuda()
                cover_img_B = cover_img_B.cuda()
                secret_img = secret_img.cuda()
                clean_img = clean_img.cuda()
                copyright_img = copyright_img.cuda()

            concat_img = torch.cat([cover_img_B, secret_img], dim=1)
            concat_imgv = Variable(concat_img)
            cover_imgv = Variable(cover_img_B)
            secret_imgv = Variable(secret_img)
            clean_imgv = Variable(clean_img)
            copyright_imgv = Variable(copyright_img)
            A_imgv = Variable(cover_img_A)
            container_img = Hnet(concat_imgv)
            remover_concate_img = torch.cat([container_img, clean_imgv], dim=1)
            remover_concate_imgv = Variable(remover_concate_img)
            origin_rev_secret_img = Enet(container_img)


            uncover_img = Remover(remover_concate_imgv)
            rev_secret_img = Enet(uncover_img) 
            
            # N, C, H, W = rev_secret_img.shape

            # # diag_elements = torch.ones(H)
            # diag_elements = 0.00000001 + (torch.rand(H) * 0.00000009)
            # M = torch.diag(diag_elements).view(1, 1, H, H).expand(N, 1, H, H)
            # I = torch.eye(H).view(1, 1, H, H).expand(N, 1, H, H)
            # M_1 = M + I
            
            # # M = M.view(1, 1, H, W).expand_as(rev_secret_img)
            # # M_1 = M_1.view(1, 1, H, W).expand_as(rev_secret_img)
            
            # if torch.cuda.is_available():
            #     M = M.cuda()
            #     M_1 = M_1.cuda()
                
            # trans_rev_secret_img = M_1 @ secret_img - M @ rev_secret_img
            nc = calculate_nc_tensor(rev_secret_img, secret_img)
            print(nc)
            if nc > 0.9:
                ans = secret_imgv
            else:
                ans = clean_imgv
            half_batchsize = int(this_batch_size / 2)
            err_mse = mse_loss(ans, clean_imgv)
            err_origin = mse_loss(rev_secret_img, clean_imgv)
            # err_consist = opt.betamse * mse_loss(trans_rev_secret_img[0:half_batchsize, :, :, :],
            #                                       trans_rev_secret_img[half_batchsize:this_batch_size, :, :, :])
            pixel_loss = mse_loss(uncover_img, container_img)
            err_remover = opt.betamse * err_mse  + opt.betamse * pixel_loss

            RemoverMselosses.update(err_mse.data, this_batch_size)
            RemoverOriginlosses.update(err_origin.data, this_batch_size)
            RemoverPixellosses.update(pixel_loss.data, this_batch_size)
            # RemoverConsistlosses.update(err_consist.data, this_batch_size)
            Removerlosses.update(err_remover.data, this_batch_size)

            if i % 50 == 0:
                save_result_pic(this_batch_size, cover_img_A, cover_imgv.data, container_img.data, uncover_img.data,
                                secret_img, origin_rev_secret_img.data, rev_secret_img.data, ans.data,  
                               epoch, i, opt.validationpics)

    val_time = time.time() - start_time
    val_log = "validation[%d] val_mseloss = %.6f\t val_originloss = %.6f\t val_l1loss = %.6f\t val_consistloss = %.6f\t val_sumloss = %.6f\t validation time=%.2f" % (
        epoch,  RemoverMselosses.avg, RemoverOriginlosses.avg, RemoverPixellosses.avg, RemoverConsistlosses.avg, Removerlosses.avg, val_time)

    print_log(val_log, logPath)


    writer.add_scalar('validation/Mse_loss', RemoverMselosses.avg, epoch)
    writer.add_scalar('validation/L1_loss', RemoverPixellosses.avg, epoch)
    writer.add_scalar('validation/Consist_loss', RemoverConsistlosses.avg, epoch)
    writer.add_scalar('validation/Sum_loss', Removerlosses.avg, epoch)

    print(
        "#################################################### validation end ########################################################")

    return RemoverMselosses.avg, RemoverPixellosses.avg, RemoverConsistlosses.avg, Removerlosses.avg


# custom weights initialization
# these initializations are often used in GAN.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)

def calculate_nc_tensor(tensor1, tensor2):

    # Flatten the tensors to compute the dot product more easily.
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)
    
    # Compute the dot product of the two tensors.
    dot_product = torch.dot(tensor1_flat, tensor2_flat)
    
    # Compute the norms of the tensors.
    norm_tensor1 = torch.norm(tensor1_flat)
    norm_tensor2 = torch.norm(tensor2_flat)
    
    # Prevent division by zero
    if norm_tensor1 == 0 or norm_tensor2 == 0:
        return float('inf')
    
    # Compute the NC value.
    nc_value = dot_product / (norm_tensor1 * norm_tensor2)
    
    return nc_value.item()  # Convert to Python float for easier handling

# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)

def save_result_pic(this_batch_size, originalLabelvA, originalLabelvB, Container_allImg, uncover_img, secretLabelv, RevSecImg,
                    origin_rev_secret_img, diff, epoch, i, save_path):
    originalFramesA = originalLabelvA.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    originalFramesB = originalLabelvB.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    container_allFrames = Container_allImg.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    uncoverFrames = uncover_img.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    # mixupFrames = mixup.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    secretFrames = secretLabelv.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    revSecFrames = RevSecImg.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)
    originrevSecFrames = origin_rev_secret_img.resize_(this_batch_size, 1, opt.imageSize, opt.imageSize)

    showResult = torch.cat(
        [originalFramesA, originalFramesB,  container_allFrames, uncoverFrames,secretFrames,
         revSecFrames, originrevSecFrames, diff], 0)

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)

# print the training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()