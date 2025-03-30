# 数据预处理
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# 自定义数据集类
class FloorPlanDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['plans']
        mask = self.dataset[idx]['colors']  # 假设数据集中有mask字段

        # 确保图像只有3个通道
        if image.mode == 'RGBA':
            image = image.convert('RGB')  # 将RGBA图像转换为RGB
        if mask.mode == 'RGBA':
            mask = mask.convert('RGB')  # 将RGBA掩码转换为RGB

        if self.transform:
            image = self.transform(image)  # 应用图像变换
            mask = self.transform(mask)  # 应用掩码变换

        return image, mask

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为256x256
    transforms.ToTensor()  # 将图像转换为PyTorch张量
])

# 加载数据集
dataset = load_dataset('zimhe/pseudo-floor-plan-12k')  # 加载伪楼层平面图数据集

# 创建数据集实例
full_train_dataset = FloorPlanDataset(dataset['train'], transform=transform)

# 选择400个样本
subset_indices = list(range(200, 600))  # 选择索引200到599的样本
train_dataset = Subset(full_train_dataset, subset_indices)

# 创建数据加载器，批量大小为4，打乱数据
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)