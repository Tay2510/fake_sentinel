import time
import copy
import torch
from tqdm import tqdm


def train_model(model, dataloaders, device, train_criterion, val_criterion, optimizer, save_path, num_epochs=10):
    history = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        since = time.time()

        phase_loss = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                progress = tqdm(iter(dataloaders[phase]), leave=False, total=len(dataloaders[phase]))
            else:
                model.eval()  # Set model to evaluate mode
                progress = iter(dataloaders[phase])

            running_loss = 0.0
            sample_counts = 0
            epoch_loss = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(progress, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = torch.sigmoid(model(inputs)).squeeze()

                    loss = val_criterion(outputs, labels.type_as(outputs))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        train_loss = train_criterion(outputs, labels.type_as(outputs))
                        train_loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                sample_counts += inputs.size(0)

                epoch_loss = running_loss / sample_counts

                if isinstance(progress, tqdm):
                    progress.set_postfix(loss='{:.4f}'.format(epoch_loss))

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

    # record the lowest validation loss
    history['best_val_loss'] = lowest_loss

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
