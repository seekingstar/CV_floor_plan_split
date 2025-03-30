# 训练模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)  # 初始化UNet模型，并将其移动到指定设备（GPU/CPU）

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，初始学习率0.001

num_epochs = 20  # 定义训练轮数（可根据训练情况调整）

# 开始训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式（启用BatchNorm/Dropout）
    running_loss = 0.0  # 累计每个epoch的损失
    
    # 遍历训练数据集
    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()  # 梯度清零
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()  # 累计当前batch的损失值
    
    # 计算并打印epoch平均损失
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
