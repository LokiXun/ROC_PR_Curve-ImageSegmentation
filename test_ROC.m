% test_ROC.m
% 1.提取图片灰度，调整阈值》》对比标注值》混淆矩阵
% filename='log_ratio.jpg';
% 3通道化为单通道》》[x,y,z]=imread() rgb图片z=3，表示由R，g，b三张图片叠起来
% [x,y,z]>>根据z>>判断是否为灰度图

filename_truth='msk3.jpg'   %原始图像为灰度图
image_truth=imread(filename_truth);

filename_array={'log_ratio.jpg','mean_ratio.jpg','Ottawa_fuse.jpg'};
color_array={'r--','k-','b-.'}
legend_filename=[];
for i=1:length(filename_array)
    str_filename=regexp(filename_array{i},'\.','split');    %分割出文件名（方法名字）
    legend_filename{i}=strrep(str_filename{1},'_','\_');    %图例legend中显示下划线要加\
end

TPR_array={};FPR_array={};  %3个图片的FPR，TPR等存于数组中》》之后一起画图
Recall_array={};Precision_array={};
AUC_array={};PR_AUC_array={};
micro_P_array={};micro_R_array={};
micro_F1_array={}
file_no=1;
while file_no<=length(filename_array)
    image=imread(filename_array{file_no});
    [x,y,z]=size(image); %xy像素的行列，z由几个图片堆叠而成》》eg:RGB图片由R,G，B三张灰度图像堆叠
    if z==3 %若为rgb图片要转成灰度图，已经为灰度图则不需要转换
        image=mat2gray(rgb2gray(image));
    end

    TPR=[];FPR=[];Precision=[];Recall=[];
    threshold_range=0:0.001:1;
    for d=threshold_range
        test_bw=im2bw(image,d);
        truth_bw=im2bw(image_truth,d); %（0，1）范围内的值由阈值d二值化》真实值类标
        TP=sum(truth_bw==1 & test_bw==1);   %按条件寻访赋值》符合条件的==1》》直接求和
        FN=sum(truth_bw==1 & test_bw==0);
        FP=sum(truth_bw==0 & test_bw==1);
        TN=sum(truth_bw==0 & test_bw==0);
        TPR=[TPR,TP/(TP+FN)];
        FPR=[FPR,FP/(FP+TN)];
        Precision=[Precision,TP/(TP+FP)];
        Recall=[Recall,TP/(TP+FN)];
    end
    % 【注】阈值d=0:0.01:1从最小开始取（近似全部预测为正类),所以ROC的TPR，FPR数组中一开始值很大
    % 1.ROC：补（0，0）(1,1)>>对FPR排序
    L=(abs(FPR-0)<0.001 &abs(TPR-0)<0.001);
    FPR(L)=[];TPR(L)=[];
    FPR=[1,FPR,0];TPR=[1,TPR,0];
    FPR=fliplr(FPR);TPR=fliplr(TPR);    %左右交换》》逆序
    
    % 2.PR:去除（0,0),补（0，1）（1，0）
    L=(abs(Recall-0)<0.0001 &abs(Precision-0)<0.0001);
    Recall(L)=[];Precision(L)=[];   %删除（0，0）
    Recall=[1,Recall,0];Precision=[0,Precision,1];
    Recall=fliplr(Recall);Precision=fliplr(Precision);  %左右交换》》逆序
    
    %计算AUC,PR_Area:要先保证FPR从小-》大
    i=1;AUC=0;%0.9841
    while i<length(FPR)
        if FPR(i+1)-FPR(i)~=0
            S_trapezoid=(TPR(i)+TPR(i+1))*(FPR(i+1)-FPR(i))/2;  %小梯形面积
            AUC=AUC+S_trapezoid;
        end
        i=i+1;
    end   
    i=1;PR_AUC=0;  %0.8707
    while i<length(Recall)  %只要Recall从小到大即可
        if Recall(i+1)-Recall(i)~=0
            S_trapezoid=(Precision(i)+Precision(i+1))*(Recall(i+1)-Recall(i))/2;  %小梯形面积
            PR_AUC=PR_AUC+S_trapezoid;
        end
        i=i+1;
    end
    
    FPR_array{file_no}=FPR;TPR_array{file_no}=TPR;
    Recall_array{file_no}=Recall;Precision_array{file_no}=Precision;
    AUC_array{file_no}=AUC;PR_AUC_array{file_no}=PR_AUC;
    micro_P_array{file_no}=sum(Precision)/length(Precision);
    micro_R_array{file_no}=sum(Recall)/length(Recall);
    micro_F1_array{file_no}=...
        (2*micro_P_array{file_no}*micro_R_array{file_no})/(micro_P_array{file_no}+micro_R_array{file_no})
        
    file_no=file_no+1;
end

% 一、ROC曲线
h_roc=figure('Name','ROC')
hold on
for i=1:length(FPR_array)
    plot(FPR_array{i},TPR_array{i},color_array{i},'LineWidth',1.1)
    text(0.5,0.5-0.1*i,[legend_filename{i},' AUC=',num2str(AUC_array{i})])
end
xlabel('FPR'),ylabel('TPR');axis([0,1.02,0,1.02]);title('ROC:comparing 3 methods');
legend(legend_filename,'Location','best');grid on; grid minor;box on;
saveas(h_roc,'ROC_3methods.jpg')
hold off
% 二、PR曲线
h_pr=figure('Name','PR')
hold on
for i=1:length(Recall_array)
    plot(Recall_array{i},Precision_array{i},color_array{i},'LineWidth',1.1)
    text(0.5,0.5-0.1*i,[legend_filename{i},' PR\_AUC=',num2str(PR_AUC_array{i})])
end
title('PR曲线');xlabel('Recall'),ylabel('Precision');axis([0,1.02,0,1.02])
x=0:0.001:1;y=x;    %比较平衡点BP
plot(x,y,'b--');
legend([legend_filename,'y=x'],'Location','best');grid on; grid minor;box on;
saveas(h_pr,'PR_3methods.jpg')
hold off


