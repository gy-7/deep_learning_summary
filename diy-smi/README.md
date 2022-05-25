# 自己diy一个smi（System management interface）



## diy-smi 1.0

在我们训练自己深度学习模型时，我们想看GPU信息，也想看CPU和内存信息。我常用的是nvidia-smi和htop，同时要开两个窗口，并且很多参数，我其实并不想查看。我们可以根据我们想要查看硬件那些信息，来自己diy一个监控窗口。





## 使用：

:one: 首先需要安装三个包，分别是用于收集gpu信息，cpu信息，美化我们的输出窗口。

```powershell
pip install nvidia-ml-py3 -i  https://pypi.mirrors.ustc.edu.cn/simple
pip install rich -i  https://pypi.mirrors.ustc.edu.cn/simple
pip install psutil -i  https://pypi.mirrors.ustc.edu.cn/simple
```



:two: 然后运行我们的脚本就可以了

```powershell
python diy-smi.py
```



## 效果展示：

#### Ubantu terminal:

<img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220525044856_ubantu_terminal.png" alt="ubantu_terminal" style="zoom:50%;" />



#### Windows Terminal Preview:

<img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220525044859_windows_terminal_preview.png" alt="image-windows_terminal_preview" style="zoom: 85%;" />
