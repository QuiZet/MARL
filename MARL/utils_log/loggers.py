from torch.multiprocessing import Queue
from threading import Thread

class NoLogger:
    def __init__(self):
        pass

    def config(*args, **kwargs):
        pass

    def log(*args, **kwargs):
        pass

    def finish(*args, **kwargs):
        pass

    def init(*args, **kwargs):
        pass

    def Image(*args, **kwargs):
        pass

    def Histogram(*args, **kwargs):
        pass

try:
    import wandb

    class WandbDistributedLogger:
        def __init__(self, *args, **kwargs):
            self.queue = Queue()

        def log_loop(self):
            while True:
                wandb.log(self.queue.get())
    
        def config(self, *args, **kwargs):
            return wandb.config(*args, **kwargs)

        def log(self, dict):
            self.queue.put(dict)

        def finish(self, *args, **kwargs):
            return wandb.finish(*args, **kwargs)

        def init(self, *args, **kwargs):
            return wandb.init(*args, **kwargs)

        def Image(self, *args, **kwargs):
            return wandb.Image(*args, **kwargs)

        def Histogram(self, *args, **kwargs):
            return wandb.Histogram(*args, **kwargs)

except ImportError as e:
    print(f'loggers exception:{e}')
else:
    pass


try:
    import dataclasses
    from datetime import datetime
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from MARL.utils_log.decorators import run_once
    class TensorboardLogger:

        tb_writer: SummaryWriter = None

        def __init__(self):
            pass

        @classmethod
        def config(cls, *args, **kwargs):
            pass

        @classmethod
        def log(cls, *args, **kwargs):
            for d in args:
                for k, v in d.items():
                    if v is not None:
                        step = 0
                        if k in cls.step_counter:
                            step = cls.step_counter[k]
                        else:
                            cls.step_counter[k] = step
                        cls.tb_writer.add_scalar(k, v, step)
                        cls.step_counter[k] += 1
                    # expected to be an image
                    if v is None and cls.img is not None:
                        cls.tb_writer.add_image(k, cls.img, dataformats="HWC")
                        cls.img = None

        def finish(*args, **kwargs):
            pass

        @classmethod
        def init(cls, *args, **kwargs):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            cls.tb_writer = SummaryWriter(log_dir='experiments/' + kwargs['project'] + '/' + str(timestamp))
            cls.to8b = lambda x: (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)
            cls.step_counter = dict()
            cls.tb_writer.add_text('config', str(kwargs))

        @classmethod
        def Image(cls, *args, **kwargs):
            """ Collect an image (not saved yet)
            """
            for v in args:
                image = v
                if isinstance(image, torch.Tensor):
                    cls.img = image.detach().cpu()
                else:
                    img = cls.plt2arr(image)
                    # Convert the figure to numpy array, read the pixel values and reshape the array
                    img = np.frombuffer(img, dtype=np.uint8)
                    img = img.reshape(image.canvas.get_width_height()[::-1] + (3,))
                    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
                    img = img / 255.0
                    # Convert to tensord
                    image = torch.tensor(img)
                if isinstance(image, torch.Tensor):
                    cls.img = cls.to8b(image)

        @classmethod
        @run_once  # (why run_once?)
        def write_model(cls, *args, **kwargs) -> None:
            """Function that writes out the model diagram
            Args:
                model: torch nn model
                indata: input data (required to create the model)
                i.e. logger.write_model(**dict(model=model, data_in=info.action))
            """
            model = kwargs['model']
            data_in = kwargs['data_in']
            cls.tb_writer.add_graph(model, data_in)

        @classmethod
        def plt2arr(cls, fig, draw=True):
            """
            need to draw if figure is not drawn yet
            """
            if draw:
                fig.canvas.draw()
            rgb_buf = fig.canvas.tostring_rgb()
            (w,h) = fig.canvas.get_width_height()
            rgb_arr = np.frombuffer(rgb_buf, dtype=np.uint8).reshape((h,w,3))
            return rgb_arr


except ImportError as e:
    print(f'loggers exception:{e}')
else:
    pass
