# train.py

## 加载并解析超参数

### opts = TrainOptions().parse()

## coach.train()

### self.net.train()

#### self.net = HairCLIPMapper(self.opts).to(self.device)

##### self,mapper = latent_mappers.HairMapper(self.opts)

###### self.clip_model

###### self.preprocess

###### self.transform

###### self.hairstyle_cut_flag

###### self.color_cut_flag

##### self.decoder

##### self.facepool

##### self.load_weights

### self.train_dataloader加载数据集

#### self.train_dataset, self.test_dataset = self.configure_datasets()加载数据

##### self.configure_datasets()

###### 有预训练的输入，就加载预训练的pt文件

###### 没有预训练输入，就用高斯分布作为输入分布

###### LatentsDataset类

###### self.hairstyle_description_list = fd.read().splitlines()读取发型文本信息(在一个txt文件里)

###### 读取发型颜色文本信息(在控制台手工输入)

###### 加载参考发型图片

###### self.out_domain_hairstyle_img_path_list

###### 参考参考发色图片

###### self.out_domain_color_img_path_list

#### DataLoader类加载batch_size个数据进行训练
