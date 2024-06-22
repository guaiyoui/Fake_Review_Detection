%Author: Shebuti Rayana

%input:
%RankList - rank list of element (e.g. users, reviews)
%label - labels of elements (-1 spam, +1 non-spam)
%flag = 1 (plot TPR vs FPR), 0 (don't plot)

%output:
%AUC - area under curve
%TPR - true positive rate as we go down the rank list
%FPR - false positive rate as we go down the rank list

function [AUC,TPR,FPR] = ROC_AUC(RankList,label,flag)   
    roc_y = label(RankList);
    tmp1 =  cumsum(roc_y == 0);
    tmp2 = cumsum(roc_y == 1);
    FPR = cumsum(roc_y == 0)./sum(roc_y == 0);
    TPR = cumsum(roc_y == 1)./sum(roc_y == 1); 
    AUC = sum((FPR(2:length(roc_y))-FPR(1:length(roc_y)-1)).*TPR(2:length(roc_y)));
    
    %Plot
    if(flag)
        color = '-r';
        h = figure;
        set(gca,'FontSize',16);
        plot(FPR,TPR,color,'LineWidth',3);
        title(strcat('AUC = ',num2str(AUC)));
        xlabel('False Positive Rate','FontSize',16);
        ylabel('True Positive Rate','FontSize',16);
        %saveas(h,filename,'jpg');
    end
end