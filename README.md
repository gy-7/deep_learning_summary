# 目标检测知识点代码汇总


## 目前已实现：

loss_function.py

+ Detector_losses: BEC_loss, CE_loss; 
+ Proposal_losses: L1_loss(MAE loss), L2_loss(MSE loss), Smooth_L1_loss, IoU_loss, GIoU_loss, DIoU_loss, CIoU_loss, AlphaIoU_loss.

spp.py

+ SPP: [Spatial Pyramid Pooling.](https://link.springer.com/chapter/10.1007/978-3-319-10578-9_23)
+ SPPF: [Spatial Pyramid Pooling - Fast (SPPF)](https://github.com/ultralytics/yolov5/tree/v6.0)