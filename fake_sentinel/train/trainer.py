import time
import copy
import torch

from fake_sentinel.train.criteria import get_criteria


def train_model(model, dataloaders, device, save_path, num_epochs=10, is_inception=True):
    history = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = float('inf')

    criterion, optimizer = get_criteria(model)

    for epoch in range(1, num_epochs + 1):
        since = time.time()

        phase_loss = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # deep copy the model
            if phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, str(save_path))

            history[phase].append(epoch_loss)

            phase_loss[phase] = epoch_loss
            time_elapsed = time.time() - since

        print('[Epoch] {}/{}, [Loss] train: {:.4f}, val: {:.4f}, [Elapse] {:.0f}m: {:.0f}s'.format(epoch,
                                                                                                   num_epochs,
                                                                                                   phase_loss['train'],
                                                                                                   phase_loss['val'],
                                                                                                   time_elapsed // 60,
                                                                                                   time_elapsed % 60))

    print('Best val loss {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
