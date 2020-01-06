import torch
import numpy as np

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler,
        n_epochs, cuda, log_interval, writer, metrics=[], start_epoch=0):
    
    for epoch in range(0, start_epoch):
        scheduler.step()
    
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Trian stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, 
                                            optimizer, cuda,
                                            log_interval, metrics)
        message = f'Epoch: {epoch+1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}'
        for metric in metrics:
            message += f'\t{metric.name}: {metric.value()}'
        
        # Test stage
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += f'\nEpoch: {epoch+1}/{n_epochs} Validation set: Average loss: {val_loss:.4}'
        for metric in metrics:
            message += f'\t{metric.name()}: {metric.value()}'

        """ logging tensorboard """
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("valid/loss", val_loss, epoch)

        print(message)

def train_epoch(train_loader, model, loss_fn, optimizer, cuda,
                log_interval, metrics):
    
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target, _, _) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)

        if cuda:
            # めっちゃ見づらいけどすべてのデータをcuda用にしてる
            for dict_ in data:
                for d in dict_:
                    dict_[d] = dict_[d].cuda()
            if target is not None:
                target = target.cuda()
        
        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs, )
        
        loss_inputs = outputs
        if target is not None:
            target = (target, )
            loss_inputs += target
        
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()


        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message +=  '\t{}: {}'.format(metric.name(), metric.value())
            
            # print(message)
            losses = []

    total_loss /= (batch_idx +1)
    return total_loss, metrics

def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target, _, _) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data, )
            if cuda:
                # めっちゃ見づらいけどすべてのデータをcuda用にしてる
                for dict_ in data:
                    for d in dict_:
                        dict_[d] = dict_[d].cuda()
                if target is not None:
                    target = target.cuda()
            
            outputs = model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs, )
            loss_inputs = outputs
            if target is not None:
                target = (target, )
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)
    
    return val_loss, metrics