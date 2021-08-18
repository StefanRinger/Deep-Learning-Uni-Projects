import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()

        # -propagate through the network
        forward = self._model(x)

        # -calculate the loss
        loss = self._crit(forward, t.squeeze(y.float()))  # squeeze to get rid of weird 1 dimension entry in the middle

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        self._optim.step()

        # -return the loss
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        prediction = self._model(x)

        # propagate through the network and calculate the loss and predictions
        loss = self._crit(prediction, t.squeeze(y.float()))

        # return the loss and the predictions
        return loss.item(), prediction

        
    def train_epoch(self):
        # set training mode
        self._model = self._model.train()

        # iterate through the training set
        total_epoch_loss = 0.0
        for image, label in self._train_dl:

            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                image = image.cuda()
                label = label.cuda()

            # perform a training step
            total_epoch_loss += self.train_step(image, label)
        
        # calculate the average loss for the epoch and return it
        return total_epoch_loss / len(self._train_dl)


    def val_test(self):

        # set eval mode
        self._model = self._model.eval()

        # disable gradient computation
        total_evaluation_loss = 0
        self.f1_values = 0
        with t.no_grad():

            # iterate through the validation set
            for image, label in self._val_test_dl:

                # transfer the batch to the gpu if given
                if self._cuda:
                    image = image.cuda()
                    label = label.cuda()
                
                # perform a validation step
                # save the predictions and the labels for each batch
                loss, predicted_label = self.val_test_step(image, label)
                
                # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
                total_evaluation_loss += loss
                self.f1_values += f1_score(t.squeeze(label.cpu()), t.squeeze(t.nn.functional.sigmoid(predicted_label.cpu()).round()), average=None)
                
        
        # return the loss and print the calculated metrics
        print('f1 score is: ', self.f1_values)
        return total_evaluation_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        
        # create a list for the train and validation losses, and create a counter for the epoch 

        training_loss = []
        test_loss = []
        epoch_count = 0
        max_f1_value = []
        self.all_test_losses = []
        self.no_progress = 0

        while True: # placeholder. Do until would be nicer though haha
      
            print('Epoch: ', epoch_count)

            # stop by epoch number
            if epoch_count == epochs:
                print('Max f1 score reached: ', max(max_f1_value), ' at epoch: ', max_f1_value.index(max(max_f1_value)))
                break


            # train for a epoch and then calculate the loss and metrics on the validation set
            avg_epoch_train_loss = self.train_epoch()
            avg_epoch_test_loss = self.val_test()
            print('Average train loss: ', avg_epoch_train_loss, ' Average test loss: ', avg_epoch_test_loss)

            current_f1_average = (self.f1_values[0] + self.f1_values[1]) / 2   # average...
            max_f1_value.append(current_f1_average)
            print("Average f1 score: ", current_f1_average)

            
            # append the losses to the respective lists
            test_loss.append(avg_epoch_test_loss)
            training_loss.append(avg_epoch_train_loss)
            self.all_test_losses.append(avg_epoch_test_loss)
            

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            self.save_checkpoint(epoch_count)


            # check whether early stopping should be performed using the early stopping criterion and stop if so
            best_loss_index = self.all_test_losses.index(min(self.all_test_losses))
            self.no_progress = epoch_count - best_loss_index

            if self.no_progress >= self._early_stopping_patience:
                print('Max f1 score reached: ', max(max_f1_value), ' at epoch: ', max_f1_value.index(max(max_f1_value)))
                break
            epoch_count += 1


        # return the losses for both training and validation
        return training_loss, test_loss

