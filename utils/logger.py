r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch
    
class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset=None):
        # self.benchmark = dataset.benchmark
        self.nclass = 1
        self.dsc_buf = []
        self.acc_buf = []
        self.sen_buf = []
        self.spe_buf = []
        self.iou_buf = []
        self.auc_buf = []
        self.cldice_buf = []
        self.vc_buf = []

        # self.loss_buf = dict()

    def update(self,DSC,Acc,Sen,Spe,IOU,AUC,cldice,VC):
        self.dsc_buf.append(DSC)
        self.acc_buf.append(Acc)
        self.sen_buf.append(Sen)
        self.spe_buf.append(Spe)
        self.iou_buf.append(IOU)
        self.auc_buf.append(AUC)
        self.cldice_buf.append(cldice)
        self.vc_buf.append(VC)

    def compute_dsc(self):
        dsc = torch.stack(self.dsc_buf)
        dsc = dsc.mean()
        return dsc
    def compute_acc(self):
        acc = torch.stack(self.acc_buf)
        acc = acc.mean()
        return acc
    
    def compute_sen(self):
        sen = torch.stack(self.sen_buf)
        sen = sen.mean()
        return sen
    def compute_spe(self):
        spe = torch.stack(self.spe_buf)
        spe = spe.mean()
        return spe
    def compute_iou(self):
        iou = torch.stack(self.iou_buf)
        iou = iou.mean()
        return iou
    def compute_auc(self):
        auc = torch.stack(self.auc_buf)
        auc = auc.mean()
        return auc
    def compute_cldice(self):
        cldice = torch.stack(self.cldice_buf)
        cldice = cldice.mean()
        return cldice
    def compute_vc(self):
        vc = torch.stack(self.vc_buf)
        vc = vc.mean()

        return vc

    # def write_result(self, split, epoch):
    #     f1 = self.compute_f1()
    #     precision = self.compute_precision()
    #     recall = self.compute_recall()
    #     msg = '\n*** %s ' % split
    #     msg += '[@Epoch %02d] ' % epoch
        
    #     msg += 'F1: %5.2f   ' % f1
    #     msg += 'Pr: %5.2f   ' % precision
    #     msg += 'R: %5.2f   ' % recall
    #     msg += '***\n'
    #     Logger.info(msg)

    # def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
    #     if batch_idx % write_batch_idx == 0:
    #         dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #         msg = '[Time: ' + dt_ms + '] '
    #         msg += '[Epoch: %02d] ' % epoch if epoch != -1 else ''
    #         msg += '[Batch: %04d/%04d] ' % (batch_idx, datalen)
    #         f1 = self.compute_f1()
            
    #         msg += 'F1: %5.2f  |  ' % f1
    #         Logger.info(msg)

class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logname = args.logname if training else '_TEST_' + args.weight.split('/')[-2].split('.')[0] #+ logtime
        if logname == '': logname = logtime

        cls.logpath = os.path.join('logs', logname + '.log')
        # cls.benchmark = args.benchmark
        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Curvilinear Segmentation. with JTFN ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_f1(cls, model, epoch, F1, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f.\n' % (epoch, F1))

    @classmethod
    def save_model_all(cls, model, epoch, F1, Pr, R, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(cls.logpath, 'best_model_all.pt'))
        cls.info('Model saved @%d w/ val. F1: %5.2f Pr: %5.2f R: %5.2f.\n' % (epoch, F1, Pr, R))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

