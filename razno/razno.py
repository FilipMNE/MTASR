import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from scipy import signal


'''
Performance of 10 fold cross validation
Average Training Loss: 2.195     Average Test Loss: 3.624        Average Training Acc: 77.67     Average Test Acc: 93.12         
Average Training MAE: 1.40      Average Test MAE: 3.02   Average Test precision: 90.90   Average Test recall: 97.50      Average Test F1_score: 93.55
'''
#---------------------------------------------------------------------------



#---------------------------------------------------------------------------
# # foldovi za 10-fold cross validation za segmente 10 sekundi framerate 1
# foldperf = {}
# foldperf['fold1'] = {
# 'train_loss': [21.419829925484105, 6.294199993232188],
# 'test_loss': [29.19629372490777, 9.536391682094997],
# 'train_acc': [62.196063296024704, 18.27479737553068],
# 'test_acc': [37.5, 65.625],
# 'train_mae': [np.array(9.637961, dtype=np.float32), np.array(4.4637527, dtype=np.float32)],
# 'test_mae': [8.349336, 9.07618],
# 'precision': [37.5, 60.97560975609756],
# 'recall': [100.0, 23.14814814814815],
# 'F1_score': [54.54545454545455, 33.557046979865774]
# }
# foldperf['fold2'] = {
#     'train_loss': [18.57994855669464, 16.16086598631499, 8.09841566682092, 7.625639904056079, 5.960008354823776, 6.436735300427006, 5.833526662306085, 13.528278602039157, 5.619255696609185],
#     'test_loss': [25.599859767489964, 10.373218218485514, 9.938678317599827, 9.918622758653429, 9.888388633728027, 9.894996325174967, 10.581982400682238, 14.291492991977268, 10.323068194919163],
#     'train_acc': [63.817059050559635, 18.197607101505213, 48.629872636047864, 11.115399459668081, 16.634504052489383, 29.21651871864145, 49.01582400617522, 17.078348128135858, 48.880741026630645],
#     'test_acc': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 69.79166666666666, 79.16666666666666, 83.85416666666666],
#     'train_mae': [
#         np.array(6.416593, dtype=np.float32),
#         np.array(8.0980215, dtype=np.float32),
#         np.array(6.958112, dtype=np.float32),
#         np.array(6.7658563, dtype=np.float32),
#         np.array(5.121497, dtype=np.float32),
#         np.array(5.631874, dtype=np.float32),
#         np.array(5.002126, dtype=np.float32),
#         np.array(12.474168, dtype=np.float32),
#         np.array(4.810607, dtype=np.float32)
#     ],
#     'test_mae': [
#         np.array(10.436797, dtype=np.float32),
#         np.array(9.890228, dtype=np.float32),
#         np.array(9.358333, dtype=np.float32),
#         np.array(9.32834, dtype=np.float32),
#         np.array(9.317145, dtype=np.float32),
#         np.array(9.273768, dtype=np.float32),
#         np.array(10.286374, dtype=np.float32),
#         np.array(12.957145, dtype=np.float32),
#         np.array(10.288922, dtype=np.float32)
#     ],
#     'precision': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 67.8125, 76.58227848101265, 88.53754940711462],
#     'recall': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 75.34722222222221, 84.02777777777779, 77.77777777777779],
#     'F1_score': [66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 71.38157894736841, 80.13245033112582, 82.80961182994456]
# }
# foldperf['fold3'] = {
#     'train_loss': [20.774386566976286, 15.107919706979896, 8.136224034300922, 6.885325392440036, 7.073700018713071, 6.242063663126372, 5.902048887961714, 5.765375791852974],
#     'test_loss': [31.810784763760036, 12.713177045186361, 9.944260067409939, 9.82335016462538, 9.827960014343262, 9.92360750834147, 9.67687914106581, 10.49034998151991],
#     'train_acc': [62.446931686607485, 22.886916248552684, 48.629872636047864, 29.35160169818603, 48.629872636047864, 48.629872636047864, 48.629872636047864, 6.368197607101505],
#     'test_acc': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 56.94444444444444],
#     'train_mae': [
#         np.array(7.9656014, dtype=np.float32),
#         np.array(7.8720136, dtype=np.float32),
#         np.array(7.1243567, dtype=np.float32),
#         np.array(6.0252304, dtype=np.float32),
#         np.array(6.225219, dtype=np.float32),
#         np.array(5.408812, dtype=np.float32),
#         np.array(5.1002116, dtype=np.float32),
#         np.array(4.676869, dtype=np.float32)
#     ],
#     'test_mae': [
#         np.array(15.699402, dtype=np.float32),
#         np.array(11.777607, dtype=np.float32),
#         np.array(9.403501, dtype=np.float32),
#         np.array(9.397816, dtype=np.float32),
#         np.array(9.35098, dtype=np.float32),
#         np.array(9.235793, dtype=np.float32),
#         np.array(8.962512, dtype=np.float32),
#         np.array(9.891702, dtype=np.float32)
#     ],
#     'precision': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 54.95049504950495],
#     'recall': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 77.08333333333334],
#     'F1_score': [66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 66.66666666666667, 64.16184971098266]
# }
# foldperf['fold4'] = {
#     'train_loss': [16.671162299167964, 6.351686351093245, 5.748165101180842, 5.3321544682538065, 5.063962232919387, 4.326149263499696],
#     'test_loss': [24.65132979150433, 10.285470204901612, 10.46117168519555, 11.574933440843111, 11.27520038192696, 13.176918418565279],
#     'train_acc': [72.37654320987654, 57.947530864197525, 2.642746913580247, 36.998456790123456, 38.73456790123457, 49.51774691358025],
#     'test_acc': [50.174216027874564, 50.174216027874564, 60.801393728223, 61.14982578397212, 63.58885017421603, 67.77003484320558],
#     'train_mae': [
#         np.array(5.724447, dtype=np.float32),
#         np.array(5.5156336, dtype=np.float32),
#         np.array(4.6532397, dtype=np.float32),
#         np.array(4.558031, dtype=np.float32),
#         np.array(4.2944384, dtype=np.float32),
#         np.array(3.5310502, dtype=np.float32)
#     ],
#     'test_mae': [
#         np.array(10.100746, dtype=np.float32),
#         np.array(9.999864, dtype=np.float32),
#         np.array(10.124925, dtype=np.float32),
#         np.array(11.626583, dtype=np.float32),
#         np.array(11.333606, dtype=np.float32),
#         np.array(12.698741, dtype=np.float32)
#     ],
#     'precision': [50.174216027874564, 50.174216027874564, 69.0909090909091, 57.08061002178649, 58.876404494382015, 62.530413625304135],
#     'recall': [100.0, 100.0, 39.58333333333333, 90.97222222222221, 90.97222222222221, 89.23611111111111],
#     'F1_score': [66.8213457076566, 66.8213457076566, 50.33112582781457, 70.14725568942437, 71.48703956343792, 73.53361945636624]
# }
# foldperf['fold5'] = {
#     'train_loss': [17.926329496605245, 16.776962569393923, 5.830580083065997, 12.87307744245187, 11.864450469802863, 7.823923722697427, 7.875628810552776, 6.969280477013656, 6.074013801671878, 6.048257021037182],
#     'test_loss': [30.813348134358723, 11.43323040008545, 11.534639464484322, 16.35576820373535, 13.842018551296658, 13.519150416056315, 16.23850335015191, 14.573648452758789, 12.016208330790201, 11.614131609598795],
#     'train_acc': [72.32728676186801, 17.00115785411038, 26.4569664222308, 36.74257043612505, 40.13894249324585, 48.649170204554224, 47.973755306831336, 55.19104592821304, 51.73678116557314, 51.254341952913926],
#     'test_acc': [50.0, 56.076388888888886, 58.333333333333336, 66.14583333333334, 68.40277777777779, 69.27083333333334, 71.35416666666666, 72.22222222222221, 75.69444444444444, 77.08333333333334],
#     'train_mae': [
#         np.array(7.3782554, dtype=np.float32),
#         np.array(8.110053, dtype=np.float32),
#         np.array(4.9488735, dtype=np.float32),
#         np.array(11.8339615, dtype=np.float32),
#         np.array(10.85768, dtype=np.float32),
#         np.array(7.019068, dtype=np.float32),
#         np.array(6.996418, dtype=np.float32),
#         np.array(6.159246, dtype=np.float32),
#         np.array(5.2565136, dtype=np.float32),
#         np.array(5.214314, dtype=np.float32)
#     ],
#     'test_mae': [
#         np.array(10.670006, dtype=np.float32),
#         np.array(10.100312, dtype=np.float32),
#         np.array(9.974704, dtype=np.float32),
#         np.array(15.784656, dtype=np.float32),
#         np.array(12.816525, dtype=np.float32),
#         np.array(12.698159, dtype=np.float32),
#         np.array(16.008825, dtype=np.float32),
#         np.array(14.038842, dtype=np.float32),
#         np.array(11.008519, dtype=np.float32),
#         np.array(10.489295, dtype=np.float32)
#     ],
#     'precision': [50.0, 54.004576659038904, 54.65116279069767, 60.739030023094685, 65.3179190751445, 63.70370370370371, 67.03601108033241, 67.3913043478261, 71.6374269005848, 78.26086956521739],
#     'recall': [100.0, 81.94444444444444, 97.91666666666666, 91.31944444444444, 78.47222222222221, 89.58333333333334, 84.02777777777779, 86.11111111111111, 85.06944444444444, 75.0],
#     'F1_score': [66.66666666666667, 65.10344827586206, 70.14925373134326, 72.95423023578364, 71.29337539432177, 74.45887445887446, 74.57627118644069, 75.60975609756098, 77.77777777777777, 76.59574468085107]
# }
# foldperf['fold6'] = {
#     'train_loss': [17.79609352739011, 9.303144413916037, 7.090231468851434, 12.938578494586157],
#     'test_loss': [26.07878049214681, 8.254695150587294, 9.909629927741157, 15.086417939927843],
#     'train_acc': [67.38710922423775, 48.629872636047864, 17.522192203782325, 28.097259745272098],
#     'test_acc': [50.0, 50.0, 53.81944444444444, 65.97222222222221],
#     'train_mae': [
#         np.array(6.8861184, dtype=np.float32),
#         np.array(8.439394, dtype=np.float32),
#         np.array(6.117089, dtype=np.float32),
#         np.array(11.843083, dtype=np.float32)
#     ],
#     'test_mae': [
#         np.array(9.540163, dtype=np.float32),
#         np.array(8.496017, dtype=np.float32),
#         np.array(10.0087595, dtype=np.float32),
#         np.array(14.460019, dtype=np.float32)
#     ],
#     'precision': [50.0, 50.0, 100.0, 61.386138613861384],
#     'recall': [100.0, 100.0, 7.638888888888889, 86.11111111111111],
#     'F1_score': [66.66666666666667, 66.66666666666667, 14.193548387096776, 71.67630057803467]
# }
# foldperf['fold7'] = {
#     'train_loss': [
#         19.683683737324547, 8.652005460813486, 6.9729179445264515,
#         5.5880413117992145, 5.676594985585009, 6.463338658690959,
#         6.862137906940597, 7.911293316513174, 8.137289473284445
#     ],
#     'test_loss': [
#         27.21680959065755, 11.579686376783583, 10.921538141038683,
#         12.48874240451389, 13.294231308831108, 12.465309354994032,
#         11.890641530354818, 12.878444459703234, 13.317191441853842
#     ],
#     'train_acc': [
#         72.32728676186801, 48.629872636047864, 33.076032419915094,
#         19.58703203396372, 19.876495561559242, 20.841373986877652,
#         22.73253570050174, 25.6657661134697, 31.860285604013892
#     ],
#     'test_acc': [
#         50.0, 50.0, 50.0,
#         50.173611111111114, 51.21527777777778, 51.388888888888886,
#         54.861111111111114, 61.111111111111114, 61.458333333333336
#     ],
#     'train_mae': [
#         np.array(8.387745, dtype=np.float32),
#         np.array(7.551851, dtype=np.float32),
#         np.array(6.1218314, dtype=np.float32),
#         np.array(4.7789526, dtype=np.float32),
#         np.array(4.8889503, dtype=np.float32),
#         np.array(5.658951, dtype=np.float32),
#         np.array(6.0698037, dtype=np.float32),
#         np.array(7.1255226, dtype=np.float32),
#         np.array(7.360178, dtype=np.float32)
#     ],
#     'test_mae': [
#         np.array(11.712113, dtype=np.float32),
#         np.array(11.520222, dtype=np.float32),
#         np.array(11.493881, dtype=np.float32),
#         np.array(12.416557, dtype=np.float32),
#         np.array(12.855576, dtype=np.float32),
#         np.array(12.630196, dtype=np.float32),
#         np.array(12.611316, dtype=np.float32),
#         np.array(14.066185, dtype=np.float32),
#         np.array(14.462001, dtype=np.float32)
#     ],
#     'precision': [
#         50.0, 50.0, 50.0,
#         50.08695652173913, 50.61511423550088, 50.70671378091873,
#         52.56410256410257, 56.32411067193676, 56.547619047619044
#     ],
#     'recall': [
#         100.0, 100.0, 100.0,
#         100.0, 100.0, 99.65277777777779,
#         99.65277777777779, 98.95833333333334, 98.95833333333334
#     ],
#     'F1_score': [
#         66.66666666666667, 66.66666666666667, 66.66666666666667,
#         66.74391657010429, 67.21120186697785, 67.21311475409837,
#         68.82494004796163, 71.78841309823679, 71.96969696969697
#     ]
# }
# foldperf['fold8'] = {
#     'train_loss': [
#         17.683243265947013, 6.355838301926219, 4.657980924589392, 5.268554534382086
#     ],
#     'test_loss': [
#         28.084778997633194, 9.068675994873047, 12.337523142496744, 12.056194093492296
#     ],
#     'train_acc': [
#         71.34311076804323, 7.71902740254728, 14.511771516788885, 34.21458896179082
#     ],
#     'test_acc': [
#         50.0, 50.173611111111114, 54.166666666666664, 55.90277777777778
#     ],
#     'train_mae': [
#         np.array(7.3841696, dtype=np.float32),
#         np.array(5.4946456, dtype=np.float32),
#         np.array(3.6817453, dtype=np.float32),
#         np.array(4.4850388, dtype=np.float32)
#     ],
#     'test_mae': [
#         np.array(8.686077, dtype=np.float32),
#         np.array(9.438445, dtype=np.float32),
#         np.array(13.177581, dtype=np.float32),
#         np.array(12.361026, dtype=np.float32)
#     ],
#     'precision': [
#         50.0, 50.08787346221442, 55.769230769230774, 53.29457364341085
#     ],
#     'recall': [
#         100.0, 98.95833333333334, 40.27777777777778, 95.48611111111111
#     ],
#     'F1_score': [
#         66.66666666666667, 66.51108518086349, 46.7741935483871, 68.40796019900498
#     ]
# }
# foldperf['fold9'] = {
#   'train_loss': [18.329684701586082, 7.3543824137338385, 7.070106537357715, 6.5820615953204715, 5.95560887797557, 5.526418264838822, 9.383011098719619, 5.781752088731525],

#   'test_loss': [25.79832278357612, 9.782408396402994, 9.605555640326607, 9.658375951978895, 9.745087517632378, 10.403214030795628, 9.902830017937553, 11.196383158365885],

#   'train_acc': [62.446931686607485, 44.80895407178696, 46.79660362794288, 6.908529525279815, 48.629872636047864, 51.37012736395214, 18.930914704747202, 24.85526823620224],

#   'test_acc': [50.0, 50.0, 50.0, 50.0, 50.0, 52.083333333333336, 52.77777777777778, 56.94444444444444],

#   'train_mae': [np.array(6.485781, dtype=np.float32), np.array(6.4925575, dtype=np.float32), np.array(6.201534, dtype=np.float32), np.array(5.463566, dtype=np.float32), np.array(5.173537, dtype=np.float32), np.array(4.766031, dtype=np.float32), np.array(8.437187, dtype=np.float32), np.array(4.2626095, dtype=np.float32)],

#   'test_mae': [np.array(10.361333, dtype=np.float32), np.array(9.662084, dtype=np.float32), np.array(9.496749, dtype=np.float32), np.array(9.406358, dtype=np.float32), np.array(9.218273, dtype=np.float32), np.array(10.104008, dtype=np.float32), np.array(9.710787, dtype=np.float32), np.array(10.874038, dtype=np.float32)],

#   'precision': [50.0, 50.0, 50.0, float('nan'), 50.0, 53.57142857142857, 51.76211453744494, 55.58659217877096],

#   'recall': [100.0, 100.0, 100.0, 0.0, 100.0, 31.25, 81.59722222222221, 69.09722222222221],

#   'F1_score': [66.66666666666667, 66.66666666666667, 66.66666666666667, float('nan'), 66.66666666666667, 39.473684210526315, 63.342318059299195, 61.60990712074304]
# }
# foldperf['fold10'] = {
#   'train_loss': [18.305447978708376, 14.237949094915887, 6.812444643641199, 6.06311747375394, 5.820115905984034, 6.721111878424894, 5.282501819096031, 12.238760809049971, 9.724220782345853],

#   'test_loss': [27.886696073744034, 9.277013566758898, 8.87056499057346, 10.14188003540039, 9.889966328938803, 10.308079613579643, 10.530620786878798, 13.953078481886122, 14.744055218166775],

#   'train_acc': [62.65920494017754, 19.606329602470087, 48.629872636047864, 46.6036279428792, 28.077962176765723, 35.95137012736395, 52.0069471246623, 19.760710150521035, 35.623311462755694],

#   'test_acc': [50.0, 50.0, 50.0, 54.513888888888886, 57.986111111111114, 64.40972222222221, 67.1875, 77.95138888888889, 78.125],

#   'train_mae': [np.array(6.174291, dtype=np.float32), np.array(6.036152, dtype=np.float32), np.array(5.820975, dtype=np.float32), np.array(5.23038, dtype=np.float32), np.array(5.0368023, dtype=np.float32), np.array(5.9516516, dtype=np.float32), np.array(4.516918, dtype=np.float32), np.array(11.253417, dtype=np.float32), np.array(8.862276, dtype=np.float32)],

#   'test_mae': [np.array(9.720298, dtype=np.float32), np.array(8.740292, dtype=np.float32), np.array(8.38519, dtype=np.float32), np.array(9.528541, dtype=np.float32), np.array(9.329606, dtype=np.float32), np.array(9.745852, dtype=np.float32), np.array(9.781965, dtype=np.float32), np.array(13.568717, dtype=np.float32), np.array(13.728717, dtype=np.float32)],

#   'precision': [50.0, 50.0, 50.0, 65.85365853658537, 54.563492063492056, 59.810874704491724, 60.5543710021322, 71.01827676240208, 73.54651162790698],

#   'recall': [100.0, 100.0, 100.0, 18.75, 95.48611111111111, 87.84722222222221, 98.61111111111111, 94.44444444444444, 87.84722222222221],

#   'F1_score': [66.66666666666667, 66.66666666666667, 66.66666666666667, 29.189189189189186, 69.44444444444444, 71.1673699015471, 75.0330250990753, 81.0730253353204, 80.0632911392405]
# }

# import datetime
# k=10
# training_info_path = "/Volumes/externiMacNovi/UBFC-Phys/segment_10s/framerate_1/training_info"
# performance_folder = os.path.join(training_info_path, f'model_performances/train_2025_05_20')
# all_folds_performance_file = open(os.path.join(performance_folder, f'all_folds_performance.txt'), 'w')

# # Racuna prosjecne performanse svih foldova (radi procjene hiperparametara: segment_len, framerate, weight_hr, weight_pp)
# testl_f, tl_f, testa_f, ta_f, te_precision, te_recall, te_F1_score, ta_mae, te_mae = [], [], [], [], [], [], [], [], []
# for f in range(1, k + 1):
#     testa = foldperf['fold{}'.format(f)]['test_acc']
#     testa_max = np.max(testa)
#     r = np.where(testa == testa_max)
#     testa_f.append(testa_max)
#     ta_f.append(foldperf['fold{}'.format(f)]['train_acc'][r[0][0]])
#     testl_f.append(foldperf['fold{}'.format(f)]['test_loss'][r[0][0]])
#     tl_f.append(foldperf['fold{}'.format(f)]['train_loss'][r[0][0]])
#     te_precision.append(foldperf['fold{}'.format(f)]['precision'][r[0][0]])
#     te_recall.append(foldperf['fold{}'.format(f)]['recall'][r[0][0]])
#     te_F1_score.append(foldperf['fold{}'.format(f)]['F1_score'][r[0][0]])
#     ta_mae.append(foldperf['fold{}'.format(f)]['train_mae'][r[0][0]])
#     te_mae.append(foldperf['fold{}'.format(f)]['test_mae'][r[0][0]])

# print('Performance of {} fold cross validation'.format(k))
# print(
#     "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f} \t Average Training MAE: {:.2f} \t Average Test MAE: {:.2f} \t Average Test precision: {:.2f} \t Average Test recall: {:.2f} \t Average Test F1_score: {:.2f}\n".format(
#         np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f), np.mean(ta_mae), np.mean(te_mae),
#         np.mean(te_precision), np.mean(te_recall), np.mean(te_F1_score)))

# all_folds_performance_file.write('Performance of {} fold cross validation\n\n'.format(k))
# all_folds_performance_file.write(
#     "Average Training Loss: {:.3f} \n Average Test Loss: {:.3f} \n Average Training Acc: {:.2f} \n Average Test Acc: {:.2f} \n Average Training MAE: {:.2f} \n Average Test MAE: {:.2f} \n Average Test precision: {:.2f} \n Average Test recall: {:.2f} \n Average Test F1_score: {:.2f}\n\n".format(
#         np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f), np.mean(ta_mae), np.mean(te_mae),
#         np.mean(te_precision), np.mean(te_recall), np.mean(te_F1_score)))

# all_folds_performance_file.close()

#---------------------------------------------------------------------------
# provjeri postoje li svi csv fajlovi u svim subject folderima
# Path to the root directory containing s1, s2, ..., s56
# base_path = "/Volumes/externiMacNovi/UBFC-Phys/segment_10s/framerate_1/rppg"

# # Expected task and segment configuration
# tasks = ["T1", "T2", "T3"]
# segments = [f"seg{str(i).zfill(2)}" for i in range(18)]

# missing_files = []

# for subject_num in range(1, 57):  # s1 to s56
#     folder_name = f"s{subject_num}"
#     folder_path = os.path.join(base_path, folder_name)
    
#     if not os.path.isdir(folder_path):
#         print(f"Folder missing: {folder_name}")
#         continue

#     for task in tasks:
#         for seg in segments:
#             expected_filename = f"rppg_s{subject_num}_{task}_{seg}.csv"
#             file_path = os.path.join(folder_path, expected_filename)
#             if not os.path.isfile(file_path):
#                 missing_files.append(file_path)

# # Report results
# if missing_files:
#     print(f"\nMissing files ({len(missing_files)} total):")
#     for file in missing_files:
#         print(file)
# else:
#     print("All expected files are present.")

#---------------------------------------------------------------------------
# ppg_signal = pd.read_csv("/Users/filipjovanovic/Desktop/OneAI/VISION/MTASR_info/rppg/s1/rppg_png_s1_T1.csv", header=None)
# ppg_signal = ppg_signal.to_numpy().squeeze(-1)
# print(ppg_signal.shape)
# print(ppg_signal[0:7])
# bvp = signal.resample(ppg_signal, 7)
# print(bvp)
# print(type(bvp))

#---------------------------------------------------------------------------
# batch = np.array([
#                         [[True, True, True], 
#                         [False, False, False], 
#                         [True, False, True],
#                         [False, False, True]],
                  
#                         [[True, True, True], 
#                         [False, False, False], 
#                         [True, False, True],
#                         [False, False, True]]
#                   ])
# print(batch.sum(axis=(1)))

# batch1 = np.array([[35,54,23],
#                    [55,35,15],
#                    [76,84,43],
#                    [27,35,43]])
# batch2 = np.array([[5,4,3],
#                    [5,5,5],
#                    [6,4,3],
#                    [7,5,3]])
# mean_bgr = np.true_divide(batch1, batch2)
# print(mean_bgr)
# print('=====================')
# b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
# print(b, g, r)
# mean_rgb = np.stack((r, g, b), axis=1)
# print('=====================')
# print(mean_rgb)
#---------------------------------------------------------------------------
# from face_detection.FaceDetectionYolo.face_detection import read_from_path, eval_image
# # contents = read_from_path('/Volumes/externiMAC/UBFC-Phys/face_not_found/s2T3frejm_55.jpg') # ima lica
# # contents2 = read_from_path('/Volumes/externiMAC/UBFC-Phys/face_not_found/s2T3frejm_5555.jpg') # nema lica
# contents3 = read_from_path('/Volumes/externiMAC/UBFC-Phys/face_not_found/s2T3frejm_55.jpg') # mozda, jpg

# # YOLO - obrada frejma - detekcija lica
# predicted_array = eval_image(contents3)
# if isinstance(predicted_array, tuple):
#     print("nije nadjeno")
# else:
#     print("jeste nadjeno")
# print(type(predicted_array))

#---------------------------------------------------------------------------
#s20 t1: jpg: 700+ sekundi, png: 950+ sekundi, ukupna greska: 0.551
#batch size: 500 ==> Time taken to read frames: 950 seconds
#batch size: 800 ==> Time taken to read frames: 894.59 seconds
#batch size: 1000 ==> Time taken to read frames: 943 seconds
#---------------------------------------------------------------------------
# cuvaj model performances u .txt fajl
# import datetime
# model_performances_path = '/Volumes/externiMAC/UBFC-Phys/training_info/model_performances'
# performance_folder = os.path.join(model_performances_path, f'train_{datetime.date.today().strftime("%Y_%m_%d")}')
# if not os.path.exists(performance_folder):
#         os.makedirs(performance_folder)
# foldperf = {}

# for fold in range(10):
#         history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'train_mae': [], 'test_mae': [],
#                    'precision': [], 'recall': [], 'F1_score': []}
        
#         history['train_loss'].append(fold)
#         history['test_loss'].append(fold)
#         history['train_acc'].append(fold)
#         history['test_acc'].append(fold)
#         history['precision'].append(fold)
#         history['recall'].append(fold)
#         history['F1_score'].append(fold)
#         history['train_mae'].append(fold)
#         history['test_mae'].append(fold)

#         foldperf['fold{}'.format(fold + 1)] = history

#         fold_performance_file = open(os.path.join(performance_folder, f'fold_{fold+1}_.txt'), 'w')
#         for perf in history:
#                 fold_performance_file.write(f'{perf}: {history[perf]}\n')

# all_perf = open(os.path.join(performance_folder, f'all_folds.txt'), 'a')
# for fold in foldperf:
#         all_perf.write(f'FOLD {fold}:\n')
#         for perf in fold:
#                 all_perf.write(f'{perf}: {foldperf[fold][perf]}\n')
#         all_perf.write('\n')
#---------------------------------------------------------------------------
# simuliraj split
# import datetime
# from UBFC_Phys_Dataset_npy_limit import data_selected
# from sklearn.model_selection import StratifiedKFold
# training_info_path = '/Volumes/externiMAC/UBFC-Phys/training_info'

# splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
# person_list, tasks, labels = data_selected()
# person_list, tasks, labels = np.array(person_list), np.array(tasks), np.array(labels)
# trainsets = []
# for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(person_list)), labels)):
#         print('Fold {}'.format(fold + 1))
#         train_p, train_t, train_l = person_list[train_idx], tasks[train_idx], labels[train_idx]
#         val_p, val_t, val_l = person_list[val_idx], tasks[val_idx], labels[val_idx]

#         # dataset_split_file = open(os.path.join(training_info_path, f'splits/split_{datetime.date.today().strftime("%Y_%m_%d")}.txt'), 'a')
#         # dataset_split_file.write(f'\tFOLD {fold+1}:\n')
#         # dataset_split_file.write(f'Train subjects: {list(train_p)}\n')
#         # dataset_split_file.write(f'Train task: {list(train_t)}\n')
#         # dataset_split_file.write(f'Test subjects: {list(val_p)}\n')
#         # dataset_split_file.write(f'Test task: {list(val_t)}\n\n')
#         for person, task in zip(val_p, val_t):
#                 trainsets.append(f'{person}_{task}')
# trainsets = sorted(trainsets, key=lambda x: int(x.split('_')[0][1:]))
# print(trainsets)
#---------------------------------------------------------------------------
# kako izgleda ubfc_phys_new.npy
# data = np.load("ubfc_phys_new.npy", allow_pickle=True)
#---------------------------------------------------------------------------
# kreiraj i pisi u .txt fajl
# import datetime
# file_path = '/Users/filipjovanovic/Desktop/razno'
# f = open(os.path.join(file_path, f'test{datetime.date.today()}.txt'), 'a')
# list1 = ['s1', 's2', 's33', 's5', 's6', 's7', 's11', 's12', 's13', 's14', 's15', 's16', 's18', 's19']
# for i in range(5):
#         f.write(f'{str(list1)}\n')
# print(datetime.date.today().strftime("%Y_%m_%d"))
#---------------------------------------------------------------------------
# nadji razliku izmedju dva csv
# import csv

# fajl1 = open('/Volumes/externiMAC/UBFC-Phys/rppg/s56/rppg_png_s56_T1_AAA.csv')
# fajl2 = open('/Volumes/externiMAC/UBFC-Phys/rppg/s56/rppg_png_s56_T1.csv')
# csvreader1 = csv.reader(fajl1)
# csvreader2 = csv.reader(fajl2)

# diff = 0
# for el1, el2 in zip(csvreader1, csvreader2):
#     el1 = float(el1[0])
#     el2 = float(el2[0])
#     diff += abs(el1 - el2)

# print(diff)
#---------------------------------------------------------------------------
# prikazi dva csv-a na istom grafu
# import pandas as pd
# import matplotlib.pyplot as plt

# # File paths
# csv_file_1 = '/Users/filipjovanovic/Desktop/rppg_all_s20_T1_png.csv'
# csv_file_2 = '/Volumes/externiMAC/UBFC-Phys/rppg/s20/rppg_all_s20_T1.csv'

# # Load data (assuming no headers) and take only first 100 rows
# data1 = pd.read_csv(csv_file_1, header=None).head(500)
# data2 = pd.read_csv(csv_file_2, header=None).head(500)

# # # jedan graf
# # # Set up the subplots side by side
# # fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# # # Plot first dataset
# # axs[0].plot(data1[0], color='blue')
# # axs[0].set_title('CSV 1 - First 100 Rows')
# # axs[0].set_xlabel('Index')
# # axs[0].set_ylabel('Value')
# # axs[0].grid(True)

# # # Plot second dataset
# # axs[1].plot(data2[0], color='orange')
# # axs[1].set_title('CSV 2 - First 100 Rows')
# # axs[1].set_xlabel('Index')
# # axs[1].set_ylabel('Value')
# # axs[1].grid(True)

# # # Layout adjustment and show
# # plt.tight_layout()
# # plt.show()


# # odvojeni grafovi
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(data1[0], label='CSV 1', color='blue')
# plt.plot(data2[0], label='CSV 2', color='orange')
# plt.title('Comparison of Two CSV Data Series')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#---------------------------------------------------------------------------

'''
pos_rppg.py:
1. cv2.imwrite(os.path.join(dataset_root, "tmp/tmp.jpg"), image)    # save frame as temporary JPG file, svi se cuvaju u jednoj jpg
                contents = read_from_path(os.path.join(dataset_root, "tmp/tmp.jpg"))
ove dvije linije koda zamijeniti kodom iz v2
2. ukloniti pisanje na disk kada nije pronadjeno lice
3. ubaciti cijelu logku u else dio -> iskopirati iz v2

v2:
tensor([1, 1, 1, 1]) tensor([[83.8370],
        [83.3510],
        [79.4274],
        [78.4050]], grad_fn=<AddmmBackward0>) tensor([[2.7154e-03, 5.4245e-02, 1.5411e-01,  ..., 3.9910e-01, 6.7547e-02, 1.3763e-01],
        [7.4231e-02, 1.0039e-01, 1.7406e-01,  ..., 5.2934e-02, 2.5046e-01, 4.1254e-01],
        [1.0646e-01, 4.1769e-02, 8.2582e-02,  ..., 7.1728e-02, 2.6823e-03, 1.2980e-01],
        [1.7414e-02, 1.7057e-01, 2.0332e-01,  ..., 4.8361e-02, 3.1795e-04, 1.1743e-01]], grad_fn=<SliceBackward0>)
OUTPUTS: tensor([[-2.5548,  1.6788],
        [-3.0911,  2.1979],
        [-2.8227,  1.9433],
        [-2.7516,  1.8778]], grad_fn=<AddmmBackward0>)

from batches:
tensor([1, 1, 1, 1]) tensor([[81.3962],
        [81.7729],
        [78.2057],
        [76.5755]], grad_fn=<AddmmBackward0>) tensor([[1.0882e-03, 8.2514e-02, 2.4089e-01,  ..., 2.2303e-02, 1.9083e-02, 1.3169e-01],
        [1.1231e-01, 8.7277e-02, 1.3732e-01,  ..., 1.6951e-01, 1.4089e-01, 1.8062e-01],
        [2.0082e-01, 9.1269e-02, 6.3983e-02,  ..., 6.2831e-02, 1.1954e-04, 9.7628e-02],
        [1.0849e-01, 2.3080e-02, 1.1642e-01,  ..., 4.3547e-02, 9.8492e-05, 1.0394e-01]], grad_fn=<SliceBackward0>)
OUTPUTS: tensor([[-2.5603,  1.6909],
        [-2.9966,  2.1012],
        [-2.8418,  1.9592],
        [-2.5772,  1.7161]], grad_fn=<AddmmBackward0>)
'''