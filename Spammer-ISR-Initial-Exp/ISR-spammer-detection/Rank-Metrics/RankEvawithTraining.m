Result = importdata('AmazonLRF10-G1');
Label = importdata('AmazonLabel.txt');
Index = Result(:,1);
Merge = [Result Label(Index')];
% user ranking
[~,R] = sort(Result(:,2),'descend');
% 
TopRight = 0;
for t=1:1000
    if(Label(Result(R(t),1))==1)
        TopRight = TopRight + 1;
    end
    if(t==500)
        P500 = TopRight/500;
    end  
end
P1000 = TopRight/1000;
filename = 'results_Evaluation.txt';
fileID = fopen(filename,'wt');
H = Merge(:,3);
[AP,MaxF,~,~] = AveragePrecision(R,Merge(:,3),0);
fprintf(fileID,'Average Precision of SpEagle: %f \n\r',AP);
    
 % area under curve
 [AUC,~,~] = ROC_AUC(R,Merge(:,3),0);
 fprintf(fileID,'Area under curve of SpEagle: %f \n\r',AUC);
 
 %0.5½Ø¶Ï£¬¼ÆËãP-R-F
 Spam = find(Merge(:,2)>=0.5);
 SSS = Merge(Spam,3);
 SpamNum = length(SSS);
 TP = length(find(SSS==1));
 P = TP/SpamNum;
 TotalSpam = length(find(Label==1));
 R = TP/TotalSpam;
 F = 2*P*R/(P+R);