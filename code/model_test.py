import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# 在Kaggle云端保存模型
save_path = '/kaggle/working/split_model.pth'

# 确保目录存在
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 保存模型状态字典
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')

# 导入必要的库
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# 设置设备为GPU（如果可用），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化UNet模型并加载训练好的模型参数
model = UNet().to(device)
model.load_state_dict(torch.load('/kaggle/working/unet_model.pth'))  # 加载训练好的模型
model.eval()  # 将模型设置为评估模式

# 定义可视化函数，用于显示图像、真实掩码和预测掩码
def visualize(image, mask, pred_mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(make_grid(image, nrow=1).permute(1, 2, 0).numpy())  # 显示图像
    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    plt.imshow(make_grid(mask, nrow=1).permute(1, 2, 0).numpy())  # 显示真实掩码
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(make_grid(pred_mask, nrow=1).permute(1, 2, 0).numpy())  # 显示预测掩码
    plt.show()

# 从数据集中取一个样本进行验证
with torch.no_grad():  # 关闭梯度计算
    for images, masks in train_loader:
        images = images.to(device)  # 将图像移动到设备
        masks = masks.to(device)  # 将掩码移动到设备
        pred_masks = model(images)  # 进行预测
        break  # 取第一个批次的数据

# 取第一个样本进行可视化
visualize(images[0].unsqueeze(0).cpu(), masks[0].unsqueeze(0).cpu(), pred_masks[0].unsqueeze(0).cpu())
