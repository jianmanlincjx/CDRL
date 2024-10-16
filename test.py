import matplotlib.pyplot as plt
import numpy as np

# # 创建一个（28, 28）的矩阵
# matrix = np.arange(28 * 28).reshape(28, 28)

# # 创建表格
# fig, ax = plt.subplots()
# ax.axis('off')
# ax.table(cellText=matrix, loc='center', cellLoc='center', colWidths=[0.1] * 28)

# plt.savefig("28*28.png")
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# 定义空间上采样的比例
upsample_factor = 8  # 224 / 28 = 8

# 创建一个 (28,28) 的示例索引矩阵
indices_28x28 = np.arange(28*28).reshape((28, 28))

# 计算每个索引在 (224,224) 中的映射
indices_224x224 = indices_28x28 * upsample_factor

# 将映射结果展平以便于查看
flattened_indices_224x224 = indices_224x224.flatten()

# 输出映射结果的前几个元素
print(flattened_indices_224x224[:20])

# 显示映射结果的分布情况
plt.plot(flattened_indices_224x224, '.')
plt.title('(28,28) Index Mapping to (224,224)')
plt.xlabel('(28,28) Index')
plt.ylabel('(224,224) Index')

plt.savefig("Upsampled.png")
