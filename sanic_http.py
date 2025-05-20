import sys
import asyncio
import itertools
import functools
from sanic import Sanic
from sanic.response import json, text, raw
from sanic.log import logger
from sanic.exceptions import ServerError

import threading
import PIL.Image
import io
import torch
import torchvision
# 假设cyclegan模块在你的项目中
# from .cyclegan import get_pretrained_model

# 为了测试，这里创建一个模拟的模型加载函数
def get_pretrained_model(model_name, map_location):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x  # 简单返回输入，实际模型会进行风格转换
    return DummyModel()

app = Sanic(__name__)

device = torch.device('cpu')
# 我们只在任何时候运行1个推理（如果需要，可以在多个运行器之间调度）
MAX_QUEUE_SIZE = 3  # 在处理"太忙"错误之前，我们接受MAX_QUEUE_SIZE的积压
MAX_BATCH_SIZE = 2  # 我们在单个批次中最多处理MAX_BATCH_SIZE个任务
MAX_WAIT = 1        # 在运行更多输入以进行批处理之前，我们最多等待MAX_WAIT秒

class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg

class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.queue = []

        self.queue_lock = None

        self.model = get_pretrained_model(self.model_name,
                                          map_location=device)

        self.needs_processing = None

        self.needs_processing_timer = None

    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.debug("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = app.loop.call_at(self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set)

    async def process_input(self, input):
        our_task = {"done_event": asyncio.Event(),
                    "input": input,
                    "time": app.loop.time()}
        async with self.queue_lock:
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HandlingError("I'm too busy", code=503)
            self.queue.append(our_task)
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait()
        return our_task["output"]

    def run_model(self, batch):  # 在其他线程中运行
        return self.model(batch.to(device)).to('cpu')

    async def model_runner(self):
        self.queue_lock = asyncio.Lock()
        self.needs_processing = asyncio.Event()
        logger.info("started model runner for {}".format(self.model_name))
        while True:
            await self.needs_processing.wait()
            self.needs_processing.clear()
            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None
            async with self.queue_lock:
                if self.queue:
                    longest_wait = app.loop.time() - self.queue[0]["time"]
                else:  # oops
                    longest_wait = None
                logger.debug("launching processing. queue size: {}. longest wait: {}".format(len(self.queue), longest_wait))
                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[:len(to_process)]
                self.schedule_processing_if_needed()
            # 这里进行复制，如果能避免会更简洁
            batch = torch.stack([t["input"] for t in to_process], dim=0)
            # 我们可以在这里删除输入...

            result = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, batch)
            )
            for t, r in zip(to_process, result):
                t["output"] = r
                t["done_event"].set()
            del to_process

def main():
    global style_transfer_runner
    # 使用命令行参数或默认值
    model_name = sys.argv[1] if len(sys.argv) > 1 else "default_model"
    style_transfer_runner = ModelRunner(model_name)

    @app.route('/image', methods=['PUT'], stream=True)
    async def image(request):
        try:
            print(request.headers)
            content_length = int(request.headers.get('content-length', '0'))
            MAX_SIZE = 2**22 # 4MB
            if content_length:
                if content_length > MAX_SIZE:
                    raise HandlingError("Too large")
                data = bytearray(content_length)
            else:
                data = bytearray(MAX_SIZE)
            pos = 0
            while True:
                # 这里仍然有过多的数据复制
                data_part = await request.stream.read()
                if data_part is None:
                    break
                data[pos: len(data_part) + pos] = data_part
                pos += len(data_part)
                if pos > MAX_SIZE:
                    raise HandlingError("Too large")

            # 理想情况下，我们应该最小化预处理...
            im = PIL.Image.open(io.BytesIO(data))
            im = torchvision.transforms.functional.resize(im, (228, 228))
            im = torchvision.transforms.functional.to_tensor(im)
            im = im[:3]  # 如果存在alpha通道则丢弃
            if im.dim() != 3 or im.size(0) < 3 or im.size(0) > 4:
                raise HandlingError("need rgb image")
            out_im = await style_transfer_runner.process_input(im)
            out_im = torchvision.transforms.functional.to_pil_image(out_im)
            imgByteArr = io.BytesIO()
            out_im.save(imgByteArr, format='JPEG')
            return raw(imgByteArr.getvalue(), status=200,
                      content_type='image/jpeg')
        except HandlingError as e:
            # 我们不希望这些错误被记录...
            return text(e.handling_msg, status=e.handling_code)

    app.add_task(style_transfer_runner.model_runner())
    # 设置single_process=True以避免Windows上的多进程问题
    app.run(host="0.0.0.0", port=8000, debug=True, single_process=True)

if __name__ == "__main__":
    main()