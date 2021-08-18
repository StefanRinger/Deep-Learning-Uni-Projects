import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
csv_path = ''
for root, _ , files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
data_frame = pd.read_csv(csv_path, sep=';')


traingset, testset = train_test_split(data_frame, train_size=0.9, random_state=5)
y_train = traingset.iloc[0:, 1:]

# set up data loading for the training and validation set
# each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO


train_dataloader = t.utils.data.DataLoader(ChallengeDataset(traingset, 'train'), batch_size=100, shuffle=True)
test_dataloader = t.utils.data.DataLoader(ChallengeDataset(testset, 'val'), batch_size=400)

# create an instance of our ResNet model
# TODO
model = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO



crack_wt = sum(1-y_train.crack)/sum(y_train.crack)
inactive_wt = sum(1-y_train.inactive)/sum(y_train.inactive)
weight_tensor = np.asarray([crack_wt, inactive_wt]).reshape(1,2)
weight_tensor = t.as_tensor(weight_tensor)
weight_tensor = t.squeeze(weight_tensor)


loss_func = t.nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
#Optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.13)
Optimizer = t.optim.Adam(model.parameters(), lr=0.0005)
trainer_obj = Trainer(model, loss_func, Optimizer, train_dataloader, test_dataloader, True, 150)

# go, go, go... call fit on trainer
#TODO
res = trainer_obj.fit(800)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')