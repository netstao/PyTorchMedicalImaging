import matplotlib
matplotlib.use('nbagg')

import numpy as np
import matplotlib.pyplot as plt

from code1.dsets import Ct, LunaDataset

clim = (-1000.0, 300)

def findPositiveSamples(start_ndx=0, limit=100):
    ds = LunaDataset()
    positiveSample_list = []
    for sample_tup in ds.candidateInfo_list:
        if sample_tup.isNodule_bool:
            print(len(positiveSample_list), sample_tup)
            positiveSample_list.append(sample_tup)
        if len(positiveSample_list) >= limit:
            break
    return positiveSample_list
    
    
    
def showCandidate(series_uid, batch_ndx=None, **kwargs):
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.isNodule_bool]
#处理块索引信息，如果没有输入，就取正样本list的第一个作为索引；如果正样本list是空的，就设为0
    if batch_ndx is None:
        if pos_list:
            batch_ndx = pos_list[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = Ct(series_uid) #根据uid获取CT影像数据
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()

    fig = plt.figure(figsize=(30, 50))  #设置图像尺寸
#这个group信息是用来取切片位置，这里取了9个切片，你可以试试把里面的值换一下
    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]
#第一个子图 
#add_subplot的参数(3,4,9)意思是绘制在一个三行四列的子图的第9个位置。
    subplot = fig.add_subplot(len(group_list) + 2, 3, 1) #这里len(group_list)是3，也就是add_subplot(5,3,1)，绘制在5行三列的图像第1个位置
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)  #设置子图像的标题
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()): #设置子图像的轴信息
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[int(center_irc[0])], clim=clim, cmap='gray') #取整个CT数据的index信息展示，对应我们的俯视图

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:,int(center_irc[1])], clim=clim, cmap='gray')#取行信息展示，对应我们的正视图
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt. imshow(ct.hu_a[:,:,int(center_irc[2])], clim=clim, cmap='gray')#取列信息展示，对应我们的侧视图
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[ct_a.shape[0]//2], clim=clim, cmap='gray')#这个是取候选小块的信息，除2是放大？

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:,ct_a.shape[1]//2], clim=clim, cmap='gray')#候选小块的行信息
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:,:,ct_a.shape[2]//2], clim=clim, cmap='gray') #候选小块的列信息
    plt.gca().invert_yaxis()

#这里是绘制group的9张切片图
    for row, index_list in enumerate(group_list): 
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')


    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)
