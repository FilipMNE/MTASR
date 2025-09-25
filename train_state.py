import numpy as np
import os
import yaml
import datetime

yaml_file = "./setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))

# !!! prije pokretanja train_state.py, promijeni ove parametre u UBFC_Phys_Dataset_npy_limit.py !!!
# podesi root_dir, model za face detection, duzinu segmenta i framerate
root_dir = "/Volumes/stari/UBFC-Phys" # "/Volumes/externiMacNovi/UBFC-Phys" ili "/Volumes/stari/UBFC-Phys" ili "/Users/filipjovanovic/Desktop/OneAI/VISION/MTASR_rad_trening"
face_detection_model = 'faceBoxes' # "yolo" ili "faceBoxes"
segment_id = 'segment_30s' # "no_segmentation" ili "segment_10s" ili "segment_30s"
framerate_id = 'framerate_1' # "framerate_1" ili "framerate_5"

# HIPERPARAMETRI
LR = cfg['train']['LR']
weight_pp = cfg['train']['weight_pp']
weight_hr = cfg['train']['weight_hr']

training_info_path = f"{root_dir}/{face_detection_model}/{segment_id}/{framerate_id}/training_info"
splits_dir_path = os.path.join(training_info_path, 'splits')
if not os.path.exists(splits_dir_path):
    os.makedirs(splits_dir_path)
# save_model_path = cfg[segment_id][framerate_id]['save_model_path']
save_model_path = f"trained_models/{face_detection_model}/{segment_id}_{framerate_id}_pp{weight_pp}_hr{weight_hr}"
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
clip_len = cfg[segment_id]['clip_len']


device_list = cfg['train']['device_list']
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from models.MTASR import MetaStress
from UBFC_Phys_Dataset_npy_limit import rPPG_Dataset, data_selected
import random

seed = cfg['train']['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("device: ", device)

 
class P_HR_C_loss(nn.Module):
    def __init__(self):
        super(P_HR_C_loss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.hr_loss = nn.L1Loss()

    def forward(self, classify_result, classify_labels, peak_result, peak_labels, hr_result, hr_gt):
        cl_loss = self.classify_loss(classify_result, classify_labels)
        pp_loss = F.binary_cross_entropy(peak_result, peak_labels)
        hr_loss = self.hr_loss(hr_result, hr_gt)
        return cl_loss + weight_pp * pp_loss + weight_hr * hr_loss


class P_HR_loss(nn.Module):
    def __init__(self):
        super(P_HR_loss, self).__init__()
        self.hr_loss = nn.L1Loss()

    def forward(self, peak_result, peak_labels, hr_result, hr_gt):
        return F.binary_cross_entropy(peak_result, peak_labels) + self.hr_loss(hr_result, hr_gt)


def train_epoch(net, device, data_loader, criterion_PHC, criterion_PH, optimizer, net_type):
    net.train() # Set model to training mode
    train_loss, train_correct, MAE = 0.0, 0.0, 0.0
    for i, data in enumerate(data_loader):
        # makao sam self.gaze i self.pose, jer su vezane sa movement info, ne moze se izracunati bez njega
        # bvp: the main input signal, labels: ground-truth class labels (stress/no-stress), peak, HR: the expected pulse peaks and heart rate values
        labels, tasks, level, bvp, peak, vpg, vpg_peak, HR = data
        # labels, tasks, level, bvp, peak, vpg, vpg_peak, HR, gaze, pose = data
        # Move tensors to the correct device (CPU/GPU) - This ensures everything runs on the same device as the model.
        labels, tasks, bvp, peak, HR = Variable(labels).to(device), Variable(tasks).to(device), Variable(bvp).to(
            device), Variable(peak).to(device), Variable(HR).to(device) 

        optimizer.zero_grad() # clears old gradients from the previous training step
        if net_type == "hr":
            # Trenira samo na bvp!!!
            # print(bvp.shape) ----> torch.Size([8, 1, 2100]), 8 jer je 8 subjecta u test splitu
            hr, p_peak = net(bvp, net_type)
            loss = criterion_PH(p_peak, peak, hr.squeeze(-1), HR)

            # Backpropagation - This updates the model weights based on the computed loss.
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bvp.size(0)
            MAE += (hr.squeeze(-1) - HR).abs().mean()

        elif net_type == "both":
            outputs, hr, p_peak = net(bvp, net_type)
            loss = criterion_PHC(outputs, labels, p_peak, peak, hr.squeeze(-1), HR)

            # Backpropagation - This updates the model weights based on the computed loss.
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_loss += loss.item() * bvp.size(0)
            scores, predictions = torch.max(outputs.data, 1)
            train_correct += (predictions == labels).sum().item()
            MAE += (hr.squeeze(-1) - HR).abs().mean()
    if net_type == "hr":
        return train_loss, MAE / len(data_loader)
    elif net_type == "both":
        return train_loss, train_correct, MAE / len(data_loader)


def valid_epoch(net, device, data_loader, criterion_PHC, criterion_PH, net_type):
    net.eval()
    valid_loss, val_correct, MAE, TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i, data in enumerate(data_loader):
        # makao sam self.gaze i self.pose, jer su vezane sa movement info, ne moze se izracunati bez njega
        # labels, tasks, level, bvp, peak, vpg, vpg_peak, HR, gaze, pose = data
        labels, tasks, level, bvp, peak, vpg, vpg_peak, HR = data
        labels, tasks, bvp, peak, HR = Variable(labels).to(device), Variable(tasks).to(device), Variable(bvp).to(
            device), Variable(peak).to(device), Variable(HR).to(device)

        # Trenira samo na bvp!!!
        # print(bvp.shape) ----> torch.Size([8, 1, 2100])
        if net_type == "hr":
            hr, p_peak = net(bvp, net_type)
            loss = criterion_PH(p_peak, peak, hr.squeeze(-1), HR)
            valid_loss += loss.item() * bvp.size(0)

            MAE += (hr.squeeze(-1) - HR).abs().mean()
        elif net_type == "both":
            output_test, hr, p_peak = net(bvp, net_type)
            loss = criterion_PHC(output_test, labels, p_peak, peak, hr.squeeze(-1), HR)
            _, predicted = torch.max(output_test, 1)
            valid_loss += loss.item() * bvp.size(0)
            scores, predictions = torch.max(output_test.data, 1)
            val_correct += (predictions == labels).sum().item()
            MAE += (hr.squeeze(-1) - HR).abs().mean()

            # It tracks TP, TN, FP, FN, which are not used in training. These help compute precision, recall, and F1 score later.
            TP += np.sum((labels.detach().cpu().numpy() == 1) & (predicted.detach().cpu().numpy() == 1))
            TN += np.sum((labels.detach().cpu().numpy() == 0) & (predicted.detach().cpu().numpy() == 0))
            FP += np.sum((labels.detach().cpu().numpy() == 0) & (predicted.detach().cpu().numpy() == 1))
            FN += np.sum((labels.detach().cpu().numpy() == 1) & (predicted.detach().cpu().numpy() == 0))
    if net_type == "hr":
        return valid_loss, MAE / len(data_loader)
    elif net_type == "both":
        return valid_loss, val_correct, MAE / len(data_loader), TP, TN, FP, FN


hr_times = 50
epoch_whole = cfg['train']['epoch'] + hr_times
if __name__ == '__main__':
    # za cuvanje informacija o treningu - kako izgleda split i koje su performanse
    performance_folder = os.path.join(training_info_path, f'model_performances/pp{weight_pp}_hr{weight_hr}_train_{datetime.date.today().strftime("%Y_%m_%d")}')
    if not os.path.exists(performance_folder):
        os.makedirs(performance_folder)
    print("Performance folder: ", performance_folder)

    k = cfg['train']['cv_times'] # specifying how many cross-validation splits to use?
    splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    foldperf = {}
    person_list, tasks, labels = data_selected()
    person_list, tasks, labels = np.array(person_list), np.array(tasks), np.array(labels)
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)), labels)):
        '''
        Kako izgleda jedan fold: (data_selected() trenutno vraca 80 snimaka)
        u train: 72 snimka -> 36 su bez stresa, 36 sa stresom
            (ne mora biti od razlicith osoba, ali ako je ista osoba, mora biti razlicit nivo stresa -> mogu kombinacije (T1,T2) i (T1,T3), ali ne i (T2,T3))
        u test: 8 snimaka, 4 su bez stresa, 4 sa stresom
            (isto ne mora od razlicitih osoba -> vidi FOLD 1, uzet je s24_T1 i s24_T3)
        U folderu training_info/splits se nalaze svi foldovi
        '''
        print('Fold {}'.format(fold + 1))

        batch_size = cfg['train']['batch_size'] # 256
        num_workers = cfg['train']['num_workers'] # 1

        train_p, train_t, train_l = person_list[train_idx], tasks[train_idx], labels[train_idx]
        val_p, val_t, val_l = person_list[val_idx], tasks[val_idx], labels[val_idx]

        print('train subject:')
        print(list(train_p))
        print('train task:')
        print(list(train_t))

        print('test subject')
        print(list(val_p))
        print('test task:')
        print(list(val_t))

        # pisanje u fajl radi provjere kako izgleda split
        # ovo se moze uraditi u posebnoj skirpti, ne mora se pokretati trening samo zbog cuvanja splita
        dataset_split_file = open(os.path.join(splits_dir_path, f'split_{datetime.date.today().strftime("%Y_%m_%d")}.txt'), 'a')
        dataset_split_file.write(f'\nFOLD {fold+1}:\n')
        trainset = []
        for person, task in zip(train_p, train_t):
            trainset.append(f'{person}_{task}')
        trainset = sorted(trainset, key=lambda x: int(x.split('_')[0][1:]))
        dataset_split_file.write(f'Train set: {trainset}\n')
        testset = []
        for person, task in zip(val_p, val_t):
            testset.append(f'{person}_{task}')
        testset = sorted(testset, key=lambda x: int(x.split('_')[0][1:]))
        dataset_split_file.write(f'Test set: {testset}\n')
        dataset_split_file.close()

        # objasnjeno u klasi rPPG_Dataset
        train_dataset = rPPG_Dataset(train_p, train_t, train_l)
        test_dataset = rPPG_Dataset(val_p, val_t, val_l)

        # objasnjeno u info.txt
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        net = MetaStress().to(device) # objasnjeno u klasi MetaStress
        net = torch.nn.DataParallel(net) # This helps speed up training if more than one GPU is available.

        # Loss
        # criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss(gamma=2, alpha=0.9)

        # criterion_PHC is a combined loss that includes classification loss, peak detection loss, and heart rate loss.
        # criterion_PH includes only peak detection and heart rate loss.
        criterion_PHC, criterion_PH = P_HR_C_loss(), P_HR_loss() # za klasifikaciju se koristi cross-entropy loss

        # Using the Adam optimizer to adjust the weights of network 'net', using a learning rate of 0.001.
        # I'm using Adam instead of SGD because it is more effective when training noisy data like physiological signals (e.g., rPPG, BVP).
        optimizer = optim.Adam(
            net.parameters(),
            lr=LR,
        )
        # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'train_mae': [], 'test_mae': [],
                   'precision': [], 'recall': [], 'F1_score': []}

        best_perform = 0.0
        best_perform_mae = np.inf
        for epoch in range(1, epoch_whole + 1):
            # objasnjeno u info.txt zasto odvaja treniranje hr i hr+klasifikacija
            if epoch <= hr_times:
                train_loss, train_MAE = train_epoch(net, device, train_loader, criterion_PHC, criterion_PH, optimizer,
                                                    "hr")
                test_loss, test_MAE = valid_epoch(net, device, test_loader, criterion_PHC, criterion_PH, "hr")
                print(
                    "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training MAE {:.2f} AVG Test MAE {:.2f} ".format(
                        epoch,
                        epoch_whole,
                        train_loss / len(train_loader.sampler),
                        test_loss / len(test_loader.sampler),
                        train_MAE,
                        test_MAE
                    ))
            else:
                train_loss, train_correct, train_MAE = train_epoch(net, device, train_loader, criterion_PHC,
                                                                   criterion_PH,
                                                                   optimizer, "both")
                test_loss, test_correct, test_MAE, TP, TN, FP, FN = valid_epoch(net, device, test_loader, criterion_PHC,
                                                                                criterion_PH, "both")

                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_correct / len(train_loader.sampler) * 100
                test_loss = test_loss / len(test_loader.sampler)
                test_acc = test_correct / len(test_loader.sampler) * 100

                test_precision = TP / (TP + FP) * 100
                test_recall = TP / (TP + FN) * 100
                test_F1_score = 2 * test_precision * test_recall / (test_precision + test_recall)

                train_MAE, test_MAE = train_MAE.detach().cpu().numpy(), test_MAE.detach().cpu().numpy()

                if cfg['train']['save_model'] and test_acc > best_perform:
                    history['train_loss'].append(train_loss)
                    history['test_loss'].append(test_loss)
                    history['train_acc'].append(train_acc)
                    history['test_acc'].append(test_acc)
                    history['precision'].append(test_precision)
                    history['recall'].append(test_recall)
                    history['F1_score'].append(test_F1_score)
                    history['train_mae'].append(train_MAE)
                    history['test_mae'].append(test_MAE)
                    best_perform = test_acc
                    best_perform_mae = test_MAE
                    torch.save(net.module, os.path.join(
                        save_model_path, 'MTASR_pa{2}_hr{3}_{0}_{1}.pth'.format(fold + 1, clip_len, weight_pp, weight_hr)))
                elif cfg['train']['save_model'] and test_acc == best_perform and best_perform_mae >= test_MAE:
                    history['train_loss'].append(train_loss)
                    history['test_loss'].append(test_loss)
                    history['train_acc'].append(train_acc)
                    history['test_acc'].append(test_acc)
                    history['precision'].append(test_precision)
                    history['recall'].append(test_recall)
                    history['F1_score'].append(test_F1_score)
                    history['train_mae'].append(train_MAE)
                    history['test_mae'].append(test_MAE)
                    best_perform_mae = test_MAE
                    torch.save(net.module, os.path.join(save_model_path,
                        'MTASR_pa{2}_hr{3}_{0}_{1}.pth'.format(fold + 1, clip_len, weight_pp, weight_hr)))

                print(
                    "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} % AVG Training MAE {:.2f} AVG Test MAE {:.2f}  precision {:.2f} % recall {:.2f} % F1_score {:.2f} %".format(
                        epoch,
                        epoch_whole,
                        train_loss,
                        test_loss,
                        train_acc,
                        test_acc,
                        train_MAE,
                        test_MAE,
                        test_precision,
                        test_recall,
                        test_F1_score
                    ))
                
        foldperf['fold{}'.format(fold + 1)] = history
                
        # ovdje se zavrsava jedan fold, sacuvaj performanse u fajl
        fold_performance_file = open(os.path.join(performance_folder, f'fold_{fold+1}_.txt'), 'a')
        for performance in history:
            fold_performance_file.write(f'{performance}: {history[performance]}\n\n')
        fold_performance_file.close()
    
    # ovdje se zavrsava trening svih foldova, sacuvaj sve performanse u 1 fajl, zbog lakseg citanja svih foldova odjednom
    all_folds_performance_file = open(os.path.join(performance_folder, f'all_folds_performance.txt'), 'a')
    for fold in foldperf:
        all_folds_performance_file.write(f'FOLD {fold}:\n')
        for performance in foldperf[fold]:
            all_folds_performance_file.write(f'{performance}: {foldperf[fold][performance]}\n\n')
        all_folds_performance_file.write('------------------------------------------------------------------------------------\n\n')
    
        
    # Racuna prosjecne performanse svih foldova (radi procjene hiperparametara: segment_len, framerate, weight_hr, weight_pp)
    testl_f, tl_f, testa_f, ta_f, te_precision, te_recall, te_F1_score, ta_mae, te_mae = [], [], [], [], [], [], [], [], []
    for f in range(1, k + 1):
        testa = foldperf['fold{}'.format(f)]['test_acc']
        testa_max = np.max(testa)
        r = np.where(testa == testa_max)
        testa_f.append(testa_max)
        ta_f.append(foldperf['fold{}'.format(f)]['train_acc'][r[0][0]])
        testl_f.append(foldperf['fold{}'.format(f)]['test_loss'][r[0][0]])
        tl_f.append(foldperf['fold{}'.format(f)]['train_loss'][r[0][0]])
        te_precision.append(foldperf['fold{}'.format(f)]['precision'][r[0][0]])
        te_recall.append(foldperf['fold{}'.format(f)]['recall'][r[0][0]])
        te_F1_score.append(foldperf['fold{}'.format(f)]['F1_score'][r[0][0]])
        ta_mae.append(foldperf['fold{}'.format(f)]['train_mae'][r[0][0]])
        te_mae.append(foldperf['fold{}'.format(f)]['test_mae'][r[0][0]])

    print('Performance of {} fold cross validation'.format(k))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f} \t Average Training MAE: {:.2f} \t Average Test MAE: {:.2f} \t Average Test precision: {:.2f} \t Average Test recall: {:.2f} \t Average Test F1_score: {:.2f}\n".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f), np.mean(ta_mae), np.mean(te_mae),
            np.mean(te_precision), np.mean(te_recall), np.mean(te_F1_score)))
    
    all_folds_performance_file.write('Performance of {} fold cross validation'.format(k))
    all_folds_performance_file.write(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f} \t Average Training MAE: {:.2f} \t Average Test MAE: {:.2f} \t Average Test precision: {:.2f} \t Average Test recall: {:.2f} \t Average Test F1_score: {:.2f}\n\n".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f), np.mean(ta_mae), np.mean(te_mae),
            np.mean(te_precision), np.mean(te_recall), np.mean(te_F1_score)))

    all_folds_performance_file.close()

    #
    # if cfg['train']['save_model']:
    #     torch.save(net.module,
    #                os.path.join(save_model_path, 'meta_stress.pth'))

