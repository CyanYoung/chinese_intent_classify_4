## Chinese Intent Classify 2019-1

#### 1.preprocess

prepare() 将按类文件保存的数据汇总、去重，去除停用词，统一替换地区、时间

等特殊词，word_replace() 替换同音、同义词，打乱后划分训练、验证、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.represent

add_flag() 添加 bos，sent2ind() 将每句转换为词索引并填充为相同长度

#### 4.build

通过 trm 构建分类模型、bos 代表整句特征，对编码器词特征 x 多头

线性映射得到 q、k、v，使用点积注意力得到语境向量 c、再线性映射进行降维

#### 5.classify

predict() 实时交互，输入单句、经过清洗后预测，输出所有类别的概率

#### 6.eval

计算总体的正确率、f1 值，计算各类的精确率、召回率，显示错判样本