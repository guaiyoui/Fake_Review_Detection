%Author: Shebuti Rayana

%input:
%RankList - rank list of element (e.g. users, reviews)
%label - labels of elements (-1 spam, +1 non-spam)
%flag = 1 (plot precision vs recall), 0 (don't plot)

%output:
%AP - average precision
%Precision - precision as we go down the rank list
%Recall - as we go down the rank list

function [AP,MaxF,Precision,Recall] = AveragePrecision(RankList,label,flag)
    GroundTruth = find(label == 1);
    
    m = length(RankList);
    [~,loc] = ismember(GroundTruth,RankList);
    index = sort(loc);
    Precision = [];
    Recall = [];
    Fmeasure = [];
    
    for i = 1:length(index)
        %if(index(i)<=m-31450)
        TP = i;
        FP = index(i) - i;
        FN = length(index) - i;
        TN = m - TP - FP - FN;
        
        P = TP/(TP+FP);
        R = TP/(TP+FN);
        F = 2*P*R/(P+R);
        
        Precision = [Precision;P];
        Recall = [Recall;R];
        Fmeasure = [Fmeasure;F];
        %end
    end
    
    AP = mean(Precision);
    MaxF = max(Fmeasure);
    
    if(flag)
        % AP
        color = '-r';
        h = figure;
        set(gca,'FontSize',16);
        plot(Recall,Precision,color,'LineWidth',3);
        title(strcat('AP = ',num2str(AP)));
        xlabel('Recall','FontSize',16);
        ylabel('Precision','FontSize',16);
    end
    
end