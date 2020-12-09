% test_ROC.m
% 1.��ȡͼƬ�Ҷȣ�������ֵ�����Աȱ�עֵ����������
% filename='log_ratio.jpg';
% 3ͨ����Ϊ��ͨ������[x,y,z]=imread() rgbͼƬz=3����ʾ��R��g��b����ͼƬ������
% [x,y,z]>>����z>>�ж��Ƿ�Ϊ�Ҷ�ͼ

filename_truth='msk3.jpg'   %ԭʼͼ��Ϊ�Ҷ�ͼ
image_truth=imread(filename_truth);

filename_array={'log_ratio.jpg','mean_ratio.jpg','Ottawa_fuse.jpg'};
color_array={'r--','k-','b-.'}
legend_filename=[];
for i=1:length(filename_array)
    str_filename=regexp(filename_array{i},'\.','split');    %�ָ���ļ������������֣�
    legend_filename{i}=strrep(str_filename{1},'_','\_');    %ͼ��legend����ʾ�»���Ҫ��\
end

TPR_array={};FPR_array={};  %3��ͼƬ��FPR��TPR�ȴ��������С���֮��һ��ͼ
Recall_array={};Precision_array={};
AUC_array={};PR_AUC_array={};
micro_P_array={};micro_R_array={};
micro_F1_array={}
file_no=1;
while file_no<=length(filename_array)
    image=imread(filename_array{file_no});
    [x,y,z]=size(image); %xy���ص����У�z�ɼ���ͼƬ�ѵ����ɡ���eg:RGBͼƬ��R,G��B���ŻҶ�ͼ��ѵ�
    if z==3 %��ΪrgbͼƬҪת�ɻҶ�ͼ���Ѿ�Ϊ�Ҷ�ͼ����Ҫת��
        image=mat2gray(rgb2gray(image));
    end

    TPR=[];FPR=[];Precision=[];Recall=[];
    threshold_range=0:0.001:1;
    for d=threshold_range
        test_bw=im2bw(image,d);
        truth_bw=im2bw(image_truth,d); %��0��1����Χ�ڵ�ֵ����ֵd��ֵ������ʵֵ���
        TP=sum(truth_bw==1 & test_bw==1);   %������Ѱ�ø�ֵ������������==1����ֱ�����
        FN=sum(truth_bw==1 & test_bw==0);
        FP=sum(truth_bw==0 & test_bw==1);
        TN=sum(truth_bw==0 & test_bw==0);
        TPR=[TPR,TP/(TP+FN)];
        FPR=[FPR,FP/(FP+TN)];
        Precision=[Precision,TP/(TP+FP)];
        Recall=[Recall,TP/(TP+FN)];
    end
    % ��ע����ֵd=0:0.01:1����С��ʼȡ������ȫ��Ԥ��Ϊ����),����ROC��TPR��FPR������һ��ʼֵ�ܴ�
    % 1.ROC������0��0��(1,1)>>��FPR����
    L=(abs(FPR-0)<0.001 &abs(TPR-0)<0.001);
    FPR(L)=[];TPR(L)=[];
    FPR=[1,FPR,0];TPR=[1,TPR,0];
    FPR=fliplr(FPR);TPR=fliplr(TPR);    %���ҽ�����������
    
    % 2.PR:ȥ����0,0),����0��1����1��0��
    L=(abs(Recall-0)<0.0001 &abs(Precision-0)<0.0001);
    Recall(L)=[];Precision(L)=[];   %ɾ����0��0��
    Recall=[1,Recall,0];Precision=[0,Precision,1];
    Recall=fliplr(Recall);Precision=fliplr(Precision);  %���ҽ�����������
    
    %����AUC,PR_Area:Ҫ�ȱ�֤FPR��С-����
    i=1;AUC=0;%0.9841
    while i<length(FPR)
        if FPR(i+1)-FPR(i)~=0
            S_trapezoid=(TPR(i)+TPR(i+1))*(FPR(i+1)-FPR(i))/2;  %С�������
            AUC=AUC+S_trapezoid;
        end
        i=i+1;
    end   
    i=1;PR_AUC=0;  %0.8707
    while i<length(Recall)  %ֻҪRecall��С���󼴿�
        if Recall(i+1)-Recall(i)~=0
            S_trapezoid=(Precision(i)+Precision(i+1))*(Recall(i+1)-Recall(i))/2;  %С�������
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

% һ��ROC����
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
% ����PR����
h_pr=figure('Name','PR')
hold on
for i=1:length(Recall_array)
    plot(Recall_array{i},Precision_array{i},color_array{i},'LineWidth',1.1)
    text(0.5,0.5-0.1*i,[legend_filename{i},' PR\_AUC=',num2str(PR_AUC_array{i})])
end
title('PR����');xlabel('Recall'),ylabel('Precision');axis([0,1.02,0,1.02])
x=0:0.001:1;y=x;    %�Ƚ�ƽ���BP
plot(x,y,'b--');
legend([legend_filename,'y=x'],'Location','best');grid on; grid minor;box on;
saveas(h_pr,'PR_3methods.jpg')
hold off


