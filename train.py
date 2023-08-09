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
from models.ModelTrainer import ModelTrainer
from configs import opt
from utils import weight_init, Visualizer

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = opt.train_gpu_indexs
# Environment variables for distributed use

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '8888'

# cudnn
t.backends.cudnn.enabled = True
t.backends.cudnn.benchmark = True

def train(**kwargs):
    # print config
    opt.parse(kwargs)
    # start visdom
    vis = Visualizer(opt.env)

    # step1 model and optimizer
    # 
    model = UNet3D(1)
    # initial the params
    model.apply(weight_init)
    # pg = td.init_process_group('nccl', world_size=1, rank=2)
    #
    # model = t.nn.SyncBatchNorm.convert_sync_batchnorm(model, pg)

    best_score = 1e10

    # load model
    if opt.load_model_path:
        checkpoint_params = t.load(opt.load_model_path, map_location=lambda storage, loc: storage.cpu())
        model.load(checkpoint_params['state_dict'])
        # Adam opti
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        # # load adam
        #optimizer.load_state_dict(checkpoint_params['optimizer'])
        # meter
        best_score = checkpoint_params['best_score']

    else:
        
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 40
    # lr = 0.005    if 40 <= epoch < 50
    # lr = 0.0005   if epoch >= 50
    #scheduler = lr_scheduler.MultiStepLR(optimizer, [40-7, 50-7], gamma=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1)

    # step2 
    train_data = TrainDataset(opt.data_root, opt.resample_data_folder, 'train_series.txt', train=True)
    val_data = TrainDataset(opt.data_root, opt.resample_data_folder, 'val_series.txt', train=False)

    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # batch=1
    val_loader = DataLoader(val_data, 1, shuffle=False, num_workers=opt.num_workers)

    #step4 eval
    loss_meter = meter.AverageValueMeter()

    trainer = ModelTrainer(model)
    trainer = t.nn.DataParallel(trainer).cuda()

    # step5 training
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):

        loss_meter.reset()

        print('train epoch_{}'.format(epoch))

        for ii, (image_data, label_data, dose_data, distance_map, iso_doses, all_masks) in tqdm.tqdm(enumerate(train_loader)):
            image_data = image_data.cuda()
            dose_data = dose_data.cuda()
            label_data = label_data.cuda()
            distance_map = distance_map.cuda()
            iso_doses = iso_doses.cuda()

            pre_dose, loss = trainer(image_data, label_data, dose_data, distance_map, iso_doses, all_masks)

            loss = loss.mean()

            # save loss
            loss_meter.add(loss.cpu().data)

            # grad=0
            optimizer.zero_grad()
            # back pro，calculate grad
            loss.backward()
            # update params
            optimizer.step()

            if ii%opt.print_freq == opt.print_freq-1:
                # add loss and dice to visdom
                vis.plot('current_loss', loss_meter.value()[0])

        t.cuda.empty_cache()
        # calculate dice（validation）
        val_diff = val(model, val_loader)

        # add in visdom
        vis.plot_with_x('val_diff', epoch, val_diff)
        vis.plot_with_x('train_loss', epoch, loss_meter.value()[0])

        # ouput logs
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, val_diff:{val_diff}".format(
            epoch = epoch,
            lr = scheduler.get_last_lr()[0],
            loss = loss_meter.value()[0],
            val_diff = val_diff
        ))

        # update learning rate
        scheduler.step()

        # If the validation dice of the current epoch is greater than the highest dice, save the model parameter file of the current epoch


        if val_diff < best_score:
            best_score = val_diff
            # save checkpoint
            if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)

            t.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': val_diff
            }, os.path.join(opt.checkpoints_dir,'checkpoint_{}.pth'.format(epoch)))

        t.cuda.empty_cache()

def val(model, data_loader):
    '''
    dice
    :param model: model
    :param data_loader: dataload
    :return: dice （validation）
    '''
    # transfer to test
    model.eval()
    diff_meter = meter.AverageValueMeter()
    print('validating')

    # t.no_grad() # For the test phase, there is no need to set aside memory to save gradients

    with t.no_grad():
        for ii, (image_data, label_data, dose_data, distance_map) in tqdm.tqdm(enumerate(data_loader)):
            image_data = image_data.cuda()
            dose_data = dose_data.cuda()
            label_data = label_data.cuda()

            distance_map = distance_map.cuda()
            pred_dose = model.predict(image_data, label_data, distance_map)
            # Evaluate mean absolute error of 3D dose
            diff = t.sum(opt.max_dose * t.abs(dose_data - pred_dose)) \
                   / (image_data.shape[2] * image_data.shape[3] * image_data.shape[4])
            diff_meter.add(diff.data.cpu().numpy())

    # training
    model.train()
    # dice
    return diff_meter.value()[0]

if __name__ == '__main__':
    train()
