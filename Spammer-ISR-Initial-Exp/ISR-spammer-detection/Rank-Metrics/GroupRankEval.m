Group = importdata('review_number_group.txt');
Result = importdata('AmazonLRF10-G1');
Label = importdata('AmazonLabel.txt');

AA = ones(9424,1)-2; %None Group: -1
AA(Group(:,1)) = Group(:,3);

Index = Result(:,1);%+1;
Merge = [Result Label(Index') AA(Index')];

maxG = max(AA);
AverageP = [];
MaxFmeasure = [];
AUCurve = [];
GSize = [];
for gg=0:maxG
    GG = find(Merge(:,4) == gg); %Get Index
    MergeT = Merge(GG,:);
    % user ranking
    [~,R] = sort(MergeT(:,2),'descend');
    [AP,MaxF,~,~] = AveragePrecision(R,MergeT(:,3),0);
    [AUC,~,~] = ROC_AUC(R,MergeT(:,3),0);
    AverageP = [AverageP;AP];
    MaxFmeasure = [MaxFmeasure;MaxF];
    AUCurve = [AUCurve, AUC];
    GSize = [GSize, length(GG)];
end
