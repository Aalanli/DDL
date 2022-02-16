# %%
import os
import dill
import json
import pathlib
from itertools import cycle

from tqdm import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def make_equal_3D(tensor_list):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor


class Trainer:
    """The general trainer base class"""
    def __init__(
        self, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module = None, 
        optimizer: torch.optim.Optimizer = None, 
        directory: str = None, 
        metric_step: int = None, 
        checkpoint_step: int = None, 
        mixed_precision: bool = False, 
        lr_scheduler: bool = None,
        max_checkpoints: int = 5,
        config=None
    ) -> None:
    
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mixed_precison = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.dir = directory
        self.metric_step = metric_step
        self.checkpoint_step = checkpoint_step
        self.config = config

        self.max_checkpoints_saved = max_checkpoints
        self.steps = 0

        self.checkpointable = ['model', 'criterion', 'optimizer', 'scaler', 'steps']
        
        if isinstance(self.model, torch.nn.Module):
            if os.path.exists(self.dir):
                self.restore_latest_checkpoint()
            else:
                # first time loading model
                os.makedirs(self.dir)
                with open(self.dir + '/model_params.pkl', 'rw') as f:
                    dill.dump(config, f)
                # save a pickled referance of the objects
                # in case original class and arguments were forgotten
                with open(os.path.join(self.dir, 'obj_ref.pkl'), 'wb') as f:
                    dill.dump(self.obj_ref, f)
        elif isinstance(self.model, str):
            # if model is a string to the model directory
            self.restore_objects(os.path.join(self.model, 'obj_ref.pkl'))
            self.restore_latest_checkpoint()
    
    @property
    def obj_ref(self):
        return self.__dict__

    def restore_objects(self, file):
        with open(file, 'rb') as f:
            self.__dict__ = dill.load(f)

    def restore_checkpoint(self, file):
        checkpoint = torch.load(file)
        for k in checkpoint:
            if getattr(self.obj_ref[k], 'load_state_dict', None) is not None:
                self.obj_ref[k].load_state_dict(checkpoint[k])
            else:
                self.obj_ref[k] = checkpoint[k]
    
    def restore_latest_checkpoint(self):
        ckpt = self.get_checkpoints()
        if ckpt is not None and len(ckpt) != 0:
            self.restore_checkpoint(ckpt[-1])
            print('Restored checkpoint', ckpt[-1])
    
    def save_checkpoint(self, file):
        checkpoints = {}
        for k in self.checkpointable:
            if getattr(self.obj_ref[k], 'state_dict', None) is not None:
                checkpoints[k] = self.obj_ref[k].state_dict()
            else:
                checkpoints[k] = self.obj_ref[k]
        torch.save(checkpoints, file + '.tar')
    
    def get_checkpoints(self):
        x = sorted(pathlib.Path(self.dir).glob('**/*.tar'))
        if x is None:
            return None
        x = [str(i) for i in x]
        x = [int(os.path.basename(i)[:-4]) for i in x]  # sort by int
        x.sort()
        x = [self.dir + f'/{str(i)}.tar' for i in x]
        return x

    def regulate_checkpoints(self):
        ckpt = self.get_checkpoints()
        if ckpt is not None and len(ckpt) > self.max_checkpoints_saved:
            os.remove(ckpt[0])
        self.save_checkpoint(f'{self.dir}/{self.steps}')
    
    def save_model(self, path=None):
        if path is None:
            torch.save(self.model, f'{self.dir}/model')
        else:
            torch.save(self.model, path)
    
    def load_saved_model(self, path=None):
        """loads a model from the saved model file"""
        if path is None:
            self.model = torch.load(f'{self.dir}/model')
        else:
            self.model = torch.load(path)

    def sum_scalars(self, log: dict, new_log: dict):
        for k in log:
            log[k] += new_log[k]
    
    def div_scalars(self, log: dict, div: int):
        for k in log:
            log[k] /= div
    
    def zero_scalars(self, log: dict):
        for k in log:
            log[k] = 0
    
    def log_scalars(self, writer, log: dict, prefix=''):
        for k in log:
            writer.add_scalar(f'{prefix}/{k}', log[k] / self.metric_step, self.steps)


import wandb

class TrainerWandb(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        directory: str = None,
        metric_step: int = None,
        checkpoint_step: int = None,
        mixed_precision: bool = None,
        lr_scheduler: bool = None,
        max_checkpoints: int = None,
        max_norm=0.1,
        config=None) -> None:

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mixed_precison = mixed_precision
        self.dir = directory
        self.metric_step = metric_step
        self.checkpoint_step = checkpoint_step
        self.max_norm = max_norm
        self.config = config

        self.max_checkpoints_saved = max_checkpoints
        self.steps = 0

        self.checkpointable = ['model', 'criterion', 'optimizer', 'scaler', 'steps']
        
        self.id = self.config['name']
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

        if isinstance(self.model, torch.nn.Module):    
            if os.path.exists(self.dir):
                self.restore_latest_checkpoint()
            else:
                # first time loading model
                os.makedirs(self.dir)
                with open(self.dir + '/model_params.pkl', 'wb') as f:
                    dill.dump(config, f)
                # save a pickled referance of the objects
                # in case original class and arguments were forgotten
                with open(os.path.join(self.dir, 'obj_ref.pkl'), 'wb') as f:
                    dill.dump(self.obj_ref, f)
        elif isinstance(self.model, str):
            # if model is a string to the model directory
            self.restore_objects(os.path.join(self.model, 'obj_ref.pkl'))
            self.restore_latest_checkpoint()
        

    def train_step(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.mixed_precison):
            self.model.train()
            pred = self.model(x)
            loss_scalar = self.criterion(pred, y)
        
        self.steps += 1
        self.optimizer.zero_grad()
        self.scaler.scale(loss_scalar).backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        accr = (pred.argmax(-1) == y).float().mean()
        # disconnect from graph, for logging purposes
        loss = {'accuracy': float(accr), 'loss': float(loss_scalar)}
        return loss, self.recursive_cast(pred, {'cpu': []})
    
    def eval_step(self, x, y):
        with torch.cuda.amp.autocast(enabled=self.mixed_precison):
            with torch.no_grad():
                self.model.eval()
                self.criterion.eval()
                pred = self.model(x)
                loss_scalar = self.criterion(pred, y)

                accr = (pred.argmax(-1) == y).float().mean()
                # disconnect from graph, for logging purposes
                loss = {'accuracy': float(accr), 'loss': float(loss_scalar)}
                return loss, self.recursive_cast(pred, {'cpu': []})
    
    @staticmethod
    def call_functions(tensor, attr_args):
        """Helper function for recursive_cast"""
        for k in attr_args:
            try:
                tensor = tensor.__getattribute__(k)(*attr_args[k])
            except AttributeError:
                pass
        return tensor

    @staticmethod
    def recursive_cast(nested_tensor, attr_args):
        """Recursively casts nested tensor lists by attr_args, returns new casted results"""
        if isinstance(nested_tensor, list):
            nest = []
            for i in range(len(nested_tensor)):
                nest.append(TrainerWandb.recursive_cast(nested_tensor[i], attr_args))
            return nest
        elif isinstance(nested_tensor, dict):
            nest = {}
            for k in nested_tensor:
                nest[k] = TrainerWandb.recursive_cast(nested_tensor[k], attr_args)
            return nest
        return TrainerWandb.call_functions(nested_tensor, attr_args)
    
    def log_scalars(self, log: dict, prefix):
        prefix_copy = {f'{prefix}/{k}': v for k, v in log.items()}
        wandb.log(prefix_copy, step=self.steps)
    

    def train(self, train_data, eval_data=None):
        """train one epoch"""
        if eval_data is not None:
            eval_data = cycle(eval_data)
            eval_losses: dict = None  # placeholder

        train_losses: dict = None  # placeholder

        metric_accum = 0
        for ims, target in tqdm(train_data, desc='epoch', unit='step'):
            metric_accum += 1

            ims = make_equal_3D(ims).cuda()
            target = target.cuda()
            new_train_log, model_out = self.train_step(ims, target)
            if train_losses is None:
                train_losses = new_train_log
            else:
                # average logging across self.metric_step(s)
                self.sum_scalars(train_losses, new_train_log)
            
            if eval_data is not None:
                ims, target = next(eval_data)
                ims_eval_cu = make_equal_3D(ims).cuda()
                target_eval = self.recursive_cast(target, {'cuda': []})
                
                new_eval_log, _ = self.eval_step(ims_eval_cu, target_eval)
                if eval_losses is None:
                    eval_losses = new_eval_log
                else:
                    self.sum_scalars(eval_losses, new_eval_log)
            
            # log scalar metrics
            if (self.steps + 1) % self.metric_step == 0:
                self.div_scalars(train_losses, metric_accum)
                self.log_scalars(train_losses, 'train')
                self.zero_scalars(train_losses)
                if eval_data is not None:
                    self.div_scalars(eval_losses, metric_accum)
                    self.log_scalars(eval_losses, 'eval')
                    self.zero_scalars(eval_losses)
                metric_accum = 0
            
            if (self.steps + 1) % self.checkpoint_step == 0:
                self.regulate_checkpoints()
        self.regulate_checkpoints()  # save final checkpoint
    
    def train_epochs(self, epochs, train_data, eval_data=None, project='', entity=''):
        run = wandb.init(project=project, entity=entity, dir=self.dir, id=self.id, resume='allow', config=self.config)
        wandb.watch(self.model)
        with run:
            for i in range(epochs):
                self.train(train_data, eval_data)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
    
    def test_model(self, train_data):
        for i in range(5):
            ims, qrs, target = next(iter(train_data))

            ims_cu = self.recursive_cast(ims, {'cuda': []})
            qrs_cu = self.recursive_cast(qrs, {'cuda': []})
            target = self.recursive_cast(target, {'cuda': []})
 
            new_train_log, model_out = self.train_step((ims_cu, qrs_cu), target)
            