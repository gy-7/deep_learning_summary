import torch
import pynvml
from rich import box
from rich.console import Console
from rich.table import Table
import datetime
import psutil

LINE_LENGTH = 50
console = Console()

def prepare_mem_info():
    mem_info = psutil.virtual_memory()
    mem_used = mem_info.used / 1024 / 1024 / 1024
    mem_total = mem_info.total / 1024 / 1024 / 1024
    return mem_used, mem_total


def prepare_cpu_info(linux=True):
    cpu_name = 'Only Linux is supported '
    if linux:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.strip():
                    if line.rstrip('\n').startswith('model name'):
                        cpu_name = line.rstrip('\n').split(':')[1]
                        break
    cpu_percent = psutil.cpu_percent(interval=0.2,percpu=True)  #
    return cpu_name, cpu_percent


def prepare_gpu_info():
    # gpu_available=cuda.is_available()
    pynvml.nvmlInit()
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    diver_version = str(pynvml.nvmlSystemGetDriverVersion())[2:-1]
    version = [torch_version, cuda_version, cudnn_version, diver_version]

    gpu_num = pynvml.nvmlDeviceGetCount()
    gpu_memory_info = []
    gpu_name = []
    gpu_temperature = []
    gpu_utilrate = []
    for i in range(gpu_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_memory_info.append(pynvml.nvmlDeviceGetMemoryInfo(handle))  # memory free,total,used
        gpu_name.append("[" + str(i) + "] " + str(pynvml.nvmlDeviceGetName(handle))[2:-1] + " :rocket:")  # gpu name
        gpu_temperature.append(str(pynvml.nvmlDeviceGetTemperature(handle, 0)) + "°C")  # gpu temperature
        gpu_utilrate.append(str(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu) + "%")
    pynvml.nvmlShutdown()
    return version, gpu_num, gpu_memory_info, gpu_name, gpu_temperature, gpu_utilrate


def visiual_gpu_info(version, gpu_num, gpu_memory_info, gpu_name, gpu_temperature, gpu_utilrate):
    # console = Console()
    console.print()
    table = Table(show_footer=False)

    # data
    table.add_column("GPU Name")
    table.add_column("Memory Usage/Total", justify="right")
    table.add_column("Tempture", justify="right")
    table.add_column("UtilRate", justify="right")

    for i in range(gpu_num):
        memory_use = str(int(gpu_memory_info[i].used / 1024 / 1024)) + " / " + \
                     str(int(gpu_memory_info[i].total / 1024 / 1024)) + " MB"
        table.add_row(gpu_name[i], memory_use, gpu_temperature[i], gpu_utilrate[i])

    # cur_time = str(datetime.datetime.now())[:-7]
    # table.caption = cur_time

    table.title = ":rainbow:   " \
                  "[b][red]torch[/red][/b]:" + str(version[0]) + \
                  "    [b][red]cuda[/red][/b]:" + str(version[1]) + \
                  "    [b][red]cudnn[/red][/b]:" + str(version[2]) + \
                  "    [b][red]diver[/red][/b]:" + str(version[3]) + \
                  "   :rainbow:"

    # table style
    table.title_style = "none"
    table.caption_justify = "right"
    table.columns[0].style = "cyan"
    table.columns[0].header_style = "bold cyan"
    table.columns[1].style = "green"
    table.columns[1].header_style = "bold green"
    table.columns[2].style = "blue"
    table.columns[2].header_style = "bold blue"
    table.columns[3].style = "magenta"
    table.columns[3].header_style = "bold magenta"

    # box line
    table.box = box.SIMPLE
    console.print(table)


def visiual_mem_info(mem_used, mem_total):
    # console = Console()
    mem_use_p = int(mem_used / mem_total * LINE_LENGTH)
    mem_free_p = LINE_LENGTH - mem_use_p
    mem_line = "\[" + mem_use_p * '#' + mem_free_p * ' ' + '] '
    mem_p = str(format(mem_used, '.2f')) + "G/" + str(format(mem_total, '.2f')) + 'G'
    console.print(" [b][red]Memory :" + mem_line + mem_p)


def visiual_cpu_info(cpu_name, cpu_percent):
    # console = Console()
    console.print()
    console.print(" [b][blue]CPU name:" + cpu_name)
    for i in range(len(cpu_percent)):
        cpu_use_line = int(cpu_percent[i] / 100 * LINE_LENGTH)
        cpu_free_line = LINE_LENGTH - cpu_use_line
        cpu_line = "\[" + cpu_use_line * '#' + cpu_free_line * ' ' + '] '
        cpu_p = str(cpu_percent[i])
        cpu_p = (5 - len(cpu_p)) * " " + cpu_p
        console.print(" [b][blue]CPU [" + str(i) + "]:" + cpu_line + ' ' + cpu_p + '/100.0')
    console.print()

if __name__ == '__main__':
    version, gpu_num, gpu_memory_info, gpu_name, gpu_temperature, gpu_utilrate=prepare_gpu_info()
    mem_used, mem_total=prepare_mem_info()
    cpu_name,cpu_info=prepare_cpu_info(linux=False)

    visiual_gpu_info(version, gpu_num, gpu_memory_info, gpu_name, gpu_temperature, gpu_utilrate)
    visiual_mem_info(mem_used, mem_total)
    visiual_cpu_info(cpu_name,cpu_info)