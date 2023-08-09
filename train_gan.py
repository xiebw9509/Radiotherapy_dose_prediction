# @paper202305
import os
import torch as t
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet import meter
import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as td


from data.dataset import TrainDataset
from models.Unet3D import UNet3D
from models.Discriminator import Discriminator
from models.ModelTrainer_Gan import ModelTrainer
from configs import opt
from utils import weight_init, Visualizer

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = opt.train_gpu_indexs
# distribution
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '8888'

# cudnn
t.backends.cudnn.enabled = True
t.backends.cudnn.benchmark = True

def train(**kwargs):
    # print config
    opt.parse(kwargs)
    # start vidsom
    vis = Visualizer(opt.env)

    model_G = UNet3D(1)
    model_G.apply(weight_init)

    model_D = Discriminator()
    model_D.apply(weight_init)

    # pg = td.init_process_group('nccl', world_size=1, rank=0)
    #
    # model = t.nn.SyncBatchNorm.convert_sync_batchnorm(model, pg)

    best_score = 1e10

    
    if opt.load_model_path:
        checkpoint_params = t.load(opt.load_model_path, map_location=lambda storage, loc: storage.cpu())
        model_G.load(checkpoint_params['state_dict'])
        model_D.load(checkpoint_params['state_dict_D'])
        
        optimizer_G = optim.Adam(model_G.parameters(), lr=opt.lr)
        optimizer_D = optim.Adam(model_D.parameters(), lr=opt.lr)
        
        #optimizer.load_state_dict(checkpoint_params['optimizer'])
        
        best_score = checkpoint_params['best_score']

    else:
       
        optimizer_G = optim.Adam(model_G.parameters(), lr=opt.lr)
        optimizer_D = optim.Adam(model_D.parameters(), lr=opt.lr)

    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 40
    # lr = 0.005    if 40 <= epoch < 50
    # lr = 0.0005   if epoch >= 50
    #scheduler = lr_scheduler.MultiStepLR(optimizer, [40-7, 50-7], gamma=0.1)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)

    #
    train_data = TrainDataset(opt.data_root, opt.resample_data_folder, 'train_series.txt', train=True)
    val_data = TrainDataset(opt.data_root, opt.resample_data_folder, 'val_series.txt', train=False)

    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # batch=1
    val_loader = DataLoader(val_data, 1, shuffle=False, num_workers=opt.num_workers)

    #step4 meter
    loss_G_meter = meter.AverageValueMeter()
    loss_G_GAN_meter = meter.AverageValueMeter()
    loss_D_GAN_real_meter = meter.AverageValueMeter()
    loss_D_GAN_pred_meter = meter.AverageValueMeter()

    trainer = ModelTrainer(model_G, model_D)
    trainer = t.nn.DataParallel(trainer).cuda()

    # step5 training
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):

        loss_G_meter.reset()
        loss_G_GAN_meter.reset()
        loss_D_GAN_real_meter.reset()
        loss_D_GAN_pred_meter.reset()

        print('train epoch_{}'.format(epoch))

        for ii, (image_data, label_data, dose_data, body, distance_map) in tqdm.tqdm(enumerate(train_loader)):
            image_data = image_data.cuda()
            dose_data = dose_data.cuda()
            label_data = label_data.cuda()
            body = body.cuda()
            distance_map = distance_map.cuda()

            pre_dose, loss_G, loss_G_GAN, loss_D_real, loss_D_pred = trainer(image_data, label_data, dose_data,
                                     body, distance_map, optimizer_G, optimizer_D)

            # dice，loss
            loss_G_meter.add(loss_G.cpu().data)
            loss_G_GAN_meter.add(loss_G_GAN.cpu().data)
            loss_D_GAN_real_meter.add(loss_D_real.cpu().data)
            loss_D_GAN_pred_meter.add(loss_D_pred.cpu().data)

            if ii%opt.print_freq == opt.print_freq-1:
                # 
                vis.plot('current_loss_G', loss_G_meter.val)
                vis.plot('current_loss_G_GAN', loss_G_GAN_meter.val)
                vis.plot('current_loss_D_GAN_real', loss_D_GAN_real_meter.val)
                vis.plot('current_loss_D_GAN_pred', loss_D_GAN_pred_meter.val)

        t.cuda.empty_cache()
        # dice（validation）
        val_diff = val(model_G, val_loader)

        # visdom
        vis.plot_with_x('val_diff', epoch, val_diff)
        vis.plot_with_x('train_loss_G', epoch, loss_G_meter.value()[0])
        vis.plot_with_x('train_loss_G_GAN', epoch, loss_G_GAN_meter.value()[0])
        vis.plot_with_x('train_loss_D_GAN_real', epoch, loss_D_GAN_real_meter.value()[0])
        vis.plot_with_x('train_loss_D_GAN_pred', epoch, loss_D_GAN_pred_meter.value()[0])

        # logs
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, val_diff:{val_diff}".format(
            epoch = epoch,
            lr = opt.lr,
            loss = loss_G_meter.value()[0],
            val_diff = val_diff
        ))

        if val_diff < best_score:
            best_score = val_diff
            # save checkpoint
            if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)

            t.save({
                'epoch': epoch,
                'state_dict': model_G.state_dict(),
                'state_dict_D': model_D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'best_score': val_diff
            }, os.path.join(opt.checkpoints_dir,'checkpoint_{}.pth'.format(epoch)))

        t.cuda.empty_cache()


def val(model, data_loader):
   
    # test
    model.eval()
    diff_meter = meter.AverageValueMeter()
    print('validating')

    # t.no_grad() # For the test phase, there is no need to set aside memory to save gradients

    with t.no_grad():
        for ii, (image_data, label_data, dose_data, body, distance_map) in tqdm.tqdm(enumerate(data_loader)):
            image_data = image_data.cuda()
            dose_data = dose_data.cuda()
            label_data = label_data.cuda()
            body = body.cuda()
            distance_map = distance_map.cuda()
            pred_dose = model.predict(image_data, label_data, distance_map)
            # Evaluate mean absolute error of 3D dose
            diff = t.sum(opt.max_dose * t.abs(dose_data * body - pred_dose * body)) / t.sum(body)
            diff_meter.add(diff.data.cpu().numpy())

    # training
    model.train()
    # average dice
    return diff_meter.value()[0]

if __name__ == '__main__':
    train()


