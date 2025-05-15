import multiprocessing
import sys
import asyncio
import functools
from sanic import Sanic
from sanic.response import raw, text
from sanic.log import logger
from sanic.exceptions import ServerError
from cyclegan import get_pretrained_model
import torch
import torchvision
from PIL import Image
import io
from io import BytesIO

app = Sanic(__name__)
# 全局配置：请求超时时间（秒）
app.config.REQUEST_TIMEOUT = 600  # 5分钟
app.config.KEEP_ALIVE_TIMEOUT = 600  # 5分钟
app.config.KEEP_ALIVE  = False  # 5分钟
app.config.RESPONSE_TIMEOUT = 600  # 5分钟

# 设备配置：优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# assert torch.cuda.is_available(), "GPU not detected. Please ensure CUDA is installed."
MAX_QUEUE_SIZE = 10
MAX_BATCH_SIZE = 16
MAX_WAIT = 0.5

class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg

class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.queue = []
        self.queue_lock = asyncio.Lock()
        self.needs_processing = asyncio.Event()
        self.needs_processing_timer = None
        
        # 初始化模型（仅一次）
        self.model = self._load_model()
        self.model.eval()  # 推理模式
        self.initialized = True  # 标记初始化完成

    def _load_model(self):
        """加载模型并移动到指定设备"""
        model = get_pretrained_model(self.model_name, map_location=device)
        return model.to(device)  # 移动模型到GPU/CPU

    def schedule_processing_if_needed(self):
        if not self.queue:
            return
        
        if len(self.queue) >= MAX_BATCH_SIZE:
            self.needs_processing.set()  # 事件已提前初始化，无需检查
        else:
            # 计算下次触发时间
            next_trigger_time = self.queue[0]["time"] + MAX_WAIT
            if not self.needs_processing.is_set() and not self.needs_processing_timer:
                self.needs_processing_timer = app.loop.call_at(next_trigger_time, self.needs_processing.set)


    async def process_input(self, input_tensor):
        if self.queue_lock is None:
            raise HandlingError("Server initializing, please retry later", code=503)
        our_task = {
            "done_event": asyncio.Event(),
            "input": input_tensor,
            "time": app.loop.time(),
        }
        
        async with self.queue_lock:
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HandlingError("Server is busy", code=503)
            self.queue.append(our_task)
            self.schedule_processing_if_needed()
        
        try:
            await asyncio.wait_for(our_task["done_event"].wait(), timeout=600)
        except asyncio.TimeoutError:
            async with self.queue_lock:
                if our_task in self.queue:
                    self.queue.remove(our_task)
            raise HandlingError("Request timed out", code=504)
        
        return our_task["output"]

    def run_model(self, batch):
        """执行模型推理（在独立线程中）"""
        with torch.no_grad():  # 禁用梯度计算
            return self.model(batch).cpu()  # 推理结果返回CPU

    async def model_runner(self):
        self.needs_processing = asyncio.Event()
        logger.info(f"Model runner started for {self.model_name} on {device}")
        
        while True:
            await self.needs_processing.wait()
            self.needs_processing.clear()
            
            if self.needs_processing_timer:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None
            
            async with self.queue_lock:
                if not self.queue:
                    continue
                
                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[:len(to_process)]
                self.schedule_processing_if_needed()
            
            # 拼接批次数据
            batch = torch.stack([t["input"] for t in to_process], dim=0).to(device)
            # 在独立线程中执行推理（避免阻塞事件循环）
            result = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, batch)
            )
            
            # 分发结果
            for task, output in zip(to_process, result):
                task["output"] = output
                task["done_event"].set()


if len(sys.argv) < 2:
        raise ValueError("请指定模型路径作为参数")
style_transfer_runner = ModelRunner(sys.argv[1])
# 初始化模型 runner（确保在主进程中）
if __name__ == "__main__":
    
    app.add_task(style_transfer_runner.model_runner())

    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(style_transfer_runner.model_runner())
    
    # 启动服务（单进程模式，避免Windows多进程问题）
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
        auto_reload=False,
        access_log=True,
        workers=1  # 强制单进程
    )

@app.route("/image", methods=["PUT"], stream=True)
async def image(request):
    try:
        print (request.headers)
        content_length = int(request.headers.get('content-length', '0'))
        MAX_SIZE = 2**22 # 10MB
        if content_length:
            if content_length > MAX_SIZE:
                raise HandlingError("Too large")
            data = bytearray(content_length)
        else:
            data = bytearray(MAX_SIZE)
        pos = 0
        while True:
            # so this still copies too much stuff.
            data_part = await request.stream.read()
            if data_part is None:
                break
            data[pos: len(data_part) + pos] = data_part
            pos += len(data_part)
            print('data part size: ', len(data_part), 'total size: ', pos)
            if pos > MAX_SIZE:
                raise HandlingError("Too large")
        
        # 图像预处理
        print('data: ', len(data))
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im = torchvision.transforms.functional.resize(im, (228, 228))
        input_tensor = torchvision.transforms.functional.to_tensor(im).unsqueeze(0)  # 添加批次维度
        if not style_transfer_runner.initialized:
            raise HandlingError("Model initializing, please wait", code=503)
        # 执行推理
        print('input_tensor: ', input_tensor.shape)
        output_tensor = await style_transfer_runner.process_input(input_tensor.squeeze(0))  # 移除批次维度
        
        # 转换为图像并返回
        print('output_tensor: ', output_tensor.shape)
        output_image = torchvision.transforms.functional.to_pil_image(output_tensor)
        img_byte_arr = BytesIO()
        print('output_image: ', output_image.size)
        output_image.save(img_byte_arr, format="JPEG")
        return raw(img_byte_arr.getvalue(), content_type="image/jpeg")
    
    except HandlingError as e:
        return text(e.handling_msg, status=e.handling_code)
    except Exception as e:
        logger.exception("Error processing request")
        return text("Internal server error", status=500)