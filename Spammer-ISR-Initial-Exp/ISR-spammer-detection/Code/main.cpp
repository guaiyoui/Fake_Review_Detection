//
//  main.cpp
//  ISR2
//
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <iomanip>
#include <cctype>
#include <cstring>
#include <cassert>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
// #include </Users/zhiang/miniforge3/include/python3.9/Python.h>
#include </data1/jianweiw/env_forge/include/python3.10/Python.h>
// #include </Users/zhiang/miniforge3/include/python3.9/pythonrun.h>
#include </data1/jianweiw/env_forge/include/python3.10/pythonrun.h>

/*++++++++++++++Parameters to be Set, If Datasets are CHANGED+++++++++++++++++++++++++++++*/
#define labLen 419 //5%-419//10%-837 //30%-2512 // //Size of Labeled Users
#define unLen 9424 // Size of ALL Users
#define MAXEDGES 3000000 // Size of Network Edges
// #define labLen 833 //5%-833//10%-1667 //30%-5003 // //Size of Labeled Users
// #define unLen 16677 // Size of ALL Users
// #define MAXEDGES 10000000 // Size of Network Edges
#define ClassNum 2
static float spamclass=1; //Increase the weight of spam class
static float DD = 1;//20; // Weight d default 20
static float LL = 0;//0.05;//0.1; //Weight Lambda
static int MAXITTOTAL = 10; //20; //Maximum Number of Total Iterations
/*++++++++++++++Parameters to be Set, If Datasets are CHANGED+++++++++++++++++++++++++++++*/

#define THRESHOLD 0.5 //Spam Prob>= THRESHOLD, crisp label assigning


struct UPM {
    int uID;
    int shill;
    int tempLab;
    int neighbors[12000];
    int neighborsNum;
    float pTheta[ClassNum];  // P_\theta (y_j^{(i)=k|u_j^{(i)}})
    float pFinalLabelWeight[ClassNum]; // \hat{y_i}
    float z_jk[12000][ClassNum]; // z_{jk}^{(i)}
};
struct UPM LabUPM[labLen];
struct UPM UnUPM[unLen+1];

struct LRUP {
    int tmpLab;
    //float tmpMetric[FeatNum];
    int Data_Index; //Used in Python
    double w_jk_i;
};
struct LRUP datasetForLR[MAXEDGES + 1];

int UnLabels[unLen+1];
int Train_Index[labLen];
int Train_Label[labLen];
float Predit_Pro[unLen+1];
float alpha_k[ClassNum];  // \alpha_k

using namespace std;

void initialization() {
    // ifstream fin1("dataset/5percent/train_4.csv");//dataset/one-portion/train_2.csv
    // ifstream fin4("dataset/5percent/test_4.csv");//dataset/5percent/test_1.csv
    // ifstream fin3("dataset/jaccard0.2.txt"); // all of user neighbors
    ifstream fin1("../Data/Training_Testing/5percent/train_4.csv");//dataset/one-portion/train_2.csv
    ifstream fin4("../Data/Training_Testing/5percent/test_4.csv");//dataset/5percent/test_1.csv
    ifstream fin3("../Data/jaccard0.2.txt"); // all of user neighbors
    //ifstream fin3("network_files/UserNeighbors3.txt"); // all of user neighbors
    int i,labelID,userID;
    for (i = 1; i <= unLen; i++){
            UnUPM[i].shill = -10;
            UnLabels[i] = -10;
        }
    
    i = 0;
    while (fin1 >>userID >> labelID){   //training
        UnUPM[userID].shill = labelID;
        UnUPM[userID].tempLab = labelID;
        UnUPM[userID].uID = userID;
        UnLabels[userID] = labelID;
        Train_Index[i] = userID;
        Train_Label[i] = labelID;
        //cout<<userID<<"^^^^^^"<<labelID<<endl;
        // cout<<"labelID: "<<labelID<<"  userID:"<<userID;
        i++;
    }

        while (fin4 >>userID >> labelID){   //test
            UnUPM[userID].shill = -1;
            //fout1<<userID<<"   "<<UnUPM[userID].shill<<endl;
            UnUPM[userID].tempLab = -1;
            UnUPM[userID].uID = userID;
            UnLabels[userID] = labelID;
        }
    int cl = 0; // count of label data
    int neighbor;
    int tempUserID = 1;
    int cNeighbors = 0;
    while(fin3>>userID>>neighbor){
        if(tempUserID == userID){
            UnUPM[tempUserID].neighbors[cNeighbors] = neighbor;
            cNeighbors++;
            //if(cNeighbors>5000) cout<<"ERROR..."<<cNeighbors<<","<<userID<<endl;
        }
        else{
            UnUPM[tempUserID].neighborsNum = cNeighbors;
            tempUserID = userID;
            cNeighbors = 0;
            UnUPM[tempUserID].neighbors[cNeighbors] = neighbor;
            cNeighbors++;
        }
    }//End while (fin3>>userID>>neighbor)
    UnUPM[userID].neighborsNum = cNeighbors; //Insert the last neighbor
    
    int sum = 0;
    int N0 = 0, N1 = 0;
    int N00 = 0, N11 = 0;
    for (i = 1; i <= unLen; i++){
        sum += UnUPM[i].neighborsNum;
        if(UnLabels[i] == 1) N1+=UnUPM[i].neighborsNum;
        else if(UnLabels[i] == 0) N0+=UnUPM[i].neighborsNum;
        for(unsigned j = 0; j<UnUPM[i].neighborsNum; j++){
            if(UnLabels[i] == 1 && UnLabels[UnUPM[i].neighbors[j]] == 1) N11++;
            if(UnLabels[i] == 0 && UnLabels[UnUPM[i].neighbors[j]] == 0) N00++;
        }
    }
    cout<<"Number of Edges in Network: "<<sum<<" Total Edges of Normal:"<<N0<<" Total Edges of Spammer:"<<N1<<endl;
    cout<<"Number of Spam-Spam Edges in Network: "<<N11<<" Number of Normal-Normal Edges in Network:"<<N00<<endl;
    cout<<"Purity1: "<<(float)N11/N1<<"  Purity0:"<<(float)N00/N0<<endl;
}//End void initialization()

//*Index: Instance Index for Training; *pLabel: the corresponding pseudo-labels; *iWeight: weight of each instance; len: their length
int Python_LR_Invoke(int *Index, int *pLabel, float *iWeight, int len){
    unsigned i,j;
    // wchar_t pdir[] = L"/Users/zhiang/miniforge3";
    wchar_t pdir[] = L"/data1/jianweiw/env_forge";
    Py_SetPythonHome(pdir);
    printf("DIRDIR: %ls\n", pdir);
    Py_Initialize();
    if(!Py_IsInitialized()) {
        std::cout << "Python init failed!" << std::endl;
        return -1;
    }
    //For Checking Your Python Environment..............................
    /*cout<<"--------------CAPI ENV--------------------------"<<endl;
    printf("1. python home: %ls\n", Py_GetPythonHome());
    printf("2. program name: %ls\n", Py_GetProgramName());
    printf("3. get path: %ls\n", Py_GetPath());
    printf("4. get prefix: %ls\n", Py_GetPrefix());
    printf("5. get exec prefix: %ls\n", Py_GetExecPrefix());
    printf("6. get prog full path: %ls\n", Py_GetProgramFullPath());
    printf("7. get python version: %s\n", Py_GetVersion());
    cout<<"------------------------------------------------"<<endl;*/
    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");
    // PyRun_SimpleString("sys.path.append('/Users/zhiang/miniforge3/envs/work/lib/python3.9/site-packages')");     
    PyRun_SimpleString("sys.path.append('/data1/jianweiw/env_forge/lib/python3.10/site-packages')");     
    
    PyObject *pModule = NULL;
    PyObject *pDict = NULL;
    PyObject *pFunc = NULL;
    PyObject *pRet = NULL;
    
    pModule = PyImport_ImportModule("python2c");
    if(!pModule) {
        std::cout << "Load python2c.py failed!" << std::endl;
        return -1;
    }

    pDict = PyModule_GetDict(pModule);
    if(!pDict) {
        std::cout << "Can't find dict in python2c!" << std::endl;
        return -1;
    }

    pFunc = PyDict_GetItemString(pDict, "LR_First");
    if(!pFunc || !PyCallable_Check(pFunc)) {
        std::cout << "Can't find function!" << std::endl;
        return -1;
    }
    
    PyObject *pArgs = PyTuple_New(3);
    PyObject *pList1 = PyList_New(0);
    PyObject *pList2 = PyList_New(0);
    PyObject *pList3 = PyList_New(0);
    for(i=0; i<len; i++){
        PyList_Append(pList1, Py_BuildValue("i",Index[i]));
        PyList_Append(pList2, Py_BuildValue("i",pLabel[i]));
        PyList_Append(pList3, Py_BuildValue("f",iWeight[i]));
    } 
    PyTuple_SetItem(pArgs, 0, pList1);
    PyTuple_SetItem(pArgs, 1, pList2);
    PyTuple_SetItem(pArgs, 2, pList3);
    pRet = PyObject_CallObject(pFunc, pArgs);  
    
    //Python LR Output：Tuple->List->Element
    if(pRet == NULL) return 0;
    PyObject *pResIndex = PyTuple_GetItem(pRet,0);
    PyObject *pResSpamPro = PyTuple_GetItem(pRet,1); 
    int I1 = PyList_Size(pResIndex);
    int I2 = PyList_Size(pResSpamPro);
    cout<<I1<<"  "<<I2<<"  should be equal"<<endl;
    if (I1 != I2){
        cout<<"Return Results Error!"<<endl;
        return 0;
    }
    for(j=0; j<unLen+1; j++) Predit_Pro[j] = 0;
    for(j=0; j<I1; j++){
        int tIndex;
        float tProb;
        PyObject *E1 = PyList_GetItem(pResIndex, j);
        PyObject *E2 = PyList_GetItem(pResSpamPro, j);
        PyArg_Parse(E1, "i", &tIndex);
        PyArg_Parse(E2, "f", &tProb);
        Predit_Pro[tIndex] = tProb; //Store SPAM probability
    }
    //For Releasing the Memory, Optional.....
    //Py_DECREF(pModule);
    //Py_DECREF(pFunc);
    //Py_DECREF(pRet);
    //Py_Finalize();
    return 1;
}//End Python_LR_Invoke()

float LossFunction(float lambda, float d){
    unsigned i,j;
    float Loss=0;
    float P1,P0,PP,ww;
    
    cout<<"In Loss Function: "<<alpha_k[0]<<"..."<<alpha_k[1]<<"..."<<lambda<<endl;
    for (i = 1; i <= unLen; i++){
        // cout<<"In Loss Function start: "<<endl;
        if(UnUPM[i].pTheta[1]<0.000001) P1=0.000001;
        else P1=UnUPM[i].pTheta[1];
        if(UnUPM[i].pTheta[0]<0.000001) P0=0.000001;
        else P0=UnUPM[i].pTheta[0];
        if(UnUPM[i].shill == -1 && UnUPM[i].tempLab == 1){
            Loss -= lambda*log(P1);
        }
        else if(UnUPM[i].shill == 1){
            Loss -= log(P1);
        }
        else if(UnUPM[i].shill == -1 && UnUPM[i].tempLab == 0){
            Loss -= lambda*log(P0);
        }
        else if(UnUPM[i].shill == 0){
            Loss -= log(P0);
        }//以上计算Loss 的L1部分
        // cout<<"In Loss Function: "<<Loss<<endl;
        if(UnUPM[i].shill != -10 && UnUPM[i].neighborsNum > 0){
            ww = 1; PP = 0;
            if(UnUPM[i].shill == -1) ww = lambda;
            // cout<<"In the start: "<<endl;
            if(UnUPM[i].shill == 1 || UnUPM[i].tempLab == 1){
                for (j = 0; j < UnUPM[i].neighborsNum; j++){
                    if (UnUPM[UnUPM[i].neighbors[j]].shill == 1){
                        PP += log(alpha_k[1]);
                    }
                    else if(UnUPM[UnUPM[i].neighbors[j]].shill == 0){
                        PP += log(alpha_k[0]);
                    }
                    else if(UnUPM[UnUPM[i].neighbors[j]].shill == -1){
                        float NP1,NP0;
                        if(UnUPM[UnUPM[i].neighbors[j]].pTheta[1]<0.000001)
                            NP1 = 0.000001;
                        else NP1 = UnUPM[UnUPM[i].neighbors[j]].pTheta[1];
                        if(UnUPM[UnUPM[i].neighbors[j]].pTheta[0]<0.000001)
                            NP0 = 0.000001;
                        else NP0 = UnUPM[UnUPM[i].neighbors[j]].pTheta[0];
                        if(UnUPM[UnUPM[i].neighbors[j]].tempLab==1) PP += log(alpha_k[1] * NP1);
                        else PP += log(alpha_k[0] * NP0);
                    }
                 }//End for j
            }
            // cout<<"In the downside: "<<endl;
            if(UnUPM[i].shill == 0 || UnUPM[i].tempLab == 0){
                for (j = 0; j < UnUPM[i].neighborsNum; j++){
                    // cout<<"In the downside inner: " << UnUPM[i].uID << ", " << UnUPM[i].neighborsNum << ", " << UnUPM[i].neighbors[j] <<endl;
                    if (UnUPM[UnUPM[i].neighbors[j]].shill == 1){
                        // cout<<"In the inside 1: "<<endl;
                        PP += log(1-alpha_k[1]);
                    }
                    else if(UnUPM[UnUPM[i].neighbors[j]].shill == 0){
                        // cout<<"In the inside 2: "<<endl;
                        PP += log(1-alpha_k[0]);
                    }
                    else if(UnUPM[UnUPM[i].neighbors[j]].shill == -1){
                        // cout<<"In the inside 3: "<<endl;
                        float NP1,NP0;
                        if(UnUPM[UnUPM[i].neighbors[j]].pTheta[1]<0.000001)
                            NP1 = 0.000001;
                        else NP1 = UnUPM[UnUPM[i].neighbors[j]].pTheta[1];
                        if(UnUPM[UnUPM[i].neighbors[j]].pTheta[0]<0.000001)
                            NP0 = 0.000001;
                        else NP0 = UnUPM[UnUPM[i].neighbors[j]].pTheta[0];
                        // cout<<"In the inside 3-mid: " << log((1-alpha_k[1]) * NP1) << log((1-alpha_k[0]) * NP0) <<endl;
                        if(UnUPM[UnUPM[i].neighbors[j]].tempLab == 1) PP += log((1-alpha_k[1]) * NP1);
                        else PP += log((1-alpha_k[0]) * NP0);
                        // if(UnUPM[UnUPM[i].neighbors[j]].tempLab == 1) PP += 0;
                        // else PP += 0;
                    }
                }}
            Loss -= (d*ww/UnUPM[i].neighborsNum)*PP;
        }//End if -10
    }//End for i
    cout<<"Go out loss function...."<<endl;
    return Loss;
}//End LossFunction

float Random_Noise(){
    srand((unsigned)time(NULL));
    float rr = rand()/double(RAND_MAX);
    float rrr = (rr-0.5)*0.3;
    //for(int i=0; i<20; i++)
       // cout<<rrr<<endl;
    return rrr;
}

void InitClassifier(float lambda){
    unsigned i;
    float *iWeight = new float[labLen];
    float clsFriendsNum[ClassNum];
    clsFriendsNum[0] = 0;
    clsFriendsNum[1] = 0;
    for(i=0; i<labLen; i++)  iWeight[i] = 1;  //End 初始的instance weight 1
    Python_LR_Invoke(Train_Index, Train_Label, iWeight, labLen);

    for(i = 1; i <= unLen; i++){
        if (UnUPM[i].shill == 1){
            UnUPM[i].pTheta[1] = 1;
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1];
            UnUPM[i].pFinalLabelWeight[0]=UnUPM[i].pTheta[0];
            UnUPM[i].pFinalLabelWeight[1]=UnUPM[i].pTheta[1];
        }
        if (UnUPM[i].shill == 0){
            UnUPM[i].pTheta[1] = 0;
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1];
            UnUPM[i].pFinalLabelWeight[0]=UnUPM[i].pTheta[0];
            UnUPM[i].pFinalLabelWeight[1]=UnUPM[i].pTheta[1];
        }
        if (UnUPM[i].shill == -1){
            UnUPM[i].pTheta[1] = Predit_Pro[i];
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1];
            UnUPM[i].pFinalLabelWeight[0]=UnUPM[i].pTheta[0];
            UnUPM[i].pFinalLabelWeight[1]=UnUPM[i].pTheta[1];
        }
        if(UnUPM[i].pTheta[1] >= THRESHOLD){
            UnUPM[i].tempLab = 1;
        }
        else{
            UnUPM[i].tempLab = 0;
        }
    }
    // compute the numerator and denominator of \alpha_k
    for (i = 1; i <= unLen; i++){
        if(UnUPM[i].shill != -10 && UnUPM[i].neighborsNum > 0){
            for (int j = 0; j < UnUPM[i].neighborsNum; j++){
                if(UnUPM[i].shill == 1){
                    if (UnUPM[UnUPM[i].neighbors[j]].shill == 1)
                        clsFriendsNum[1] += 1;
                    if (UnUPM[UnUPM[i].neighbors[j]].shill == 0)
                        clsFriendsNum[0] += 1;
                    if (UnUPM[UnUPM[i].neighbors[j]].shill == -1 && UnUPM[UnUPM[i].neighbors[j]].tempLab == 1)
                        clsFriendsNum[1] += 1;
                    if (UnUPM[UnUPM[i].neighbors[j]].shill == -1 && UnUPM[UnUPM[i].neighbors[j]].tempLab == 0)
                        clsFriendsNum[0] += 1;
                }
            }
        }
    }//End Computing \alpha_k
    int tp = 0;
    int fn = 0;
    int fp = 0;
    int tn = 0;

    for (i = 1; i <= unLen; i++) {
        if (UnUPM[i].shill == -1){
            if(UnUPM[i].tempLab == 1 && UnLabels[i] == 1) tp++;
            if(UnUPM[i].tempLab == 0 && UnLabels[i] == 1) fn++;
            if(UnUPM[i].tempLab == 1 && UnLabels[i] == 0) fp++;
            if(UnUPM[i].tempLab == 0 && UnLabels[i] == 0) tn++;
        }
    }

    cout<<"tp = "<<tp<<endl;
    cout<<"fn = "<<fn<<endl;
    cout<<"fp = "<<fp<<endl;
    cout<<"tn = "<<tn<<endl;

    float recall = (float)tp/(tp+fn);
    float precision = (float)tp/(tp+fp);
    float f = 2*recall*precision /(recall + precision);

    cout<<"RECALL = "<<recall<<endl;
    cout<<"PRECISION = "<<precision<<endl;
    cout<<"F-MEASURE = "<<f<<endl;
    
    float sum = clsFriendsNum[0] + clsFriendsNum[1];
    for (int k = 0; k < ClassNum; k++)
        alpha_k[k] = (float)clsFriendsNum[k] / sum;
    cout << "alpha[" << 0 << "] = " << alpha_k[0] << endl;
    cout << "alpha[" << 1 << "] = " << alpha_k[1] << endl;
    float Loss1 = LossFunction(lambda, DD);
    cout<<"IntiLR Loss: "<<Loss1<<endl;
}//End InitClassifier(float lambda)

void ComputeAlphaK(float lambda, float d){
    float allFriendsNum[ClassNum];
    float clsFriendsNum[ClassNum];
    for (int k = 0; k < ClassNum; k++){
        clsFriendsNum[k] = 0;
        allFriendsNum[k] = 0;
    }
    for (int i = 1; i <= unLen; i++){  //
        if(UnUPM[i].shill != -10 && UnUPM[i].neighborsNum > 0){
            for (int j = 0; j < UnUPM[i].neighborsNum; j++){
                if(UnUPM[i].shill == 1 || UnUPM[i].shill == 0){
                    if(UnUPM[i].shill == 1){
                        clsFriendsNum[1] += (float)UnUPM[i].z_jk[j][1]/UnUPM[i].neighborsNum;
                        clsFriendsNum[0] += (float)UnUPM[i].z_jk[j][0]/UnUPM[i].neighborsNum;
                        allFriendsNum[0] += (float)UnUPM[i].z_jk[j][0]/UnUPM[i].neighborsNum;
                        allFriendsNum[1] += (float)UnUPM[i].z_jk[j][1]/UnUPM[i].neighborsNum;
                    }
                    else{
                        allFriendsNum[0] += (float)UnUPM[i].z_jk[j][0]/UnUPM[i].neighborsNum;
                        allFriendsNum[1] += (float)UnUPM[i].z_jk[j][1]/UnUPM[i].neighborsNum;
                    }
                }
                else{                    //need lambda
                    if(UnUPM[UnUPM[i].neighbors[j]].tempLab == 1){
                        clsFriendsNum[1] += (float)lambda * UnUPM[i].z_jk[j][1]/UnUPM[i].neighborsNum;
                        clsFriendsNum[0] += (float)lambda * UnUPM[i].z_jk[j][0]/UnUPM[i].neighborsNum;
                        allFriendsNum[0] += (float)lambda * UnUPM[i].z_jk[j][0]/UnUPM[i].neighborsNum;
                        allFriendsNum[1] += (float)lambda * UnUPM[i].z_jk[j][1]/UnUPM[i].neighborsNum;
                    }
                    else{
                        allFriendsNum[0] += (float)lambda * UnUPM[i].z_jk[j][0]/UnUPM[i].neighborsNum;
                        allFriendsNum[1] += (float)lambda * UnUPM[i].z_jk[j][1]/UnUPM[i].neighborsNum;
                    }
                }
            }
          }
  }
    // compute the numerator and denominator of \alpha_k
    for (int k = 0; k < ClassNum; k++)
        alpha_k[k] = (float)clsFriendsNum[k] / allFriendsNum[k];
    cout << "alpha[" << 0 << "] = " << alpha_k[0] << endl;
    cout << "alpha[" << 1 << "] = " << alpha_k[1] << endl;
}

void EStep_Crisp(float lambda, float d){
    cout<<"Go into EStep...."<<endl;
    unsigned i;
    for (i = 1; i <= unLen; i++){
            if (UnUPM[i].shill != -10 ){
                for (int j = 0; j < UnUPM[i].neighborsNum; j++){
                    //cout<<i<<"  "<<j<<" "<<UnUPM[i].neighborsNum<<"  "<<UnUPM[i].neighbors[j]<<"Testttt:"<<UnUPM[7].neighbors[0]<<endl;
                    if (UnUPM[UnUPM[i].neighbors[j]].shill == 0){
                        UnUPM[i].z_jk[j][0] = 1;
                        UnUPM[i].z_jk[j][1] = 0;
                    }
                    else if (UnUPM[UnUPM[i].neighbors[j]].shill == 1){
                        UnUPM[i].z_jk[j][1] = 1;
                        UnUPM[i].z_jk[j][0] = 0;
                    }
                    else if(UnUPM[UnUPM[i].neighbors[j]].shill == -1 ){
                        if(UnUPM[i].shill == 0 || UnUPM[i].tempLab == 0){ // || UnUPM[i].tempLab == 0
                            float sum = 0;
                            sum += (float)(1-alpha_k[0]) * UnUPM[UnUPM[i].neighbors[j]].pTheta[0];
                            sum += (float)(1-alpha_k[1]) * UnUPM[UnUPM[i].neighbors[j]].pTheta[1];
                            for (int k = 0; k < ClassNum; k++)
                                UnUPM[i].z_jk[j][k] = (float) ( (1-alpha_k[k]) * UnUPM[UnUPM[i].neighbors[j]].pTheta[k] ) /sum;
                        }
                        else if(UnUPM[i].shill == 1 || UnUPM[i].tempLab == 1){ // || UnUPM[i].tempLab == 0
                            float sum = 0;
                            sum += (float)alpha_k[0] * UnUPM[UnUPM[i].neighbors[j]].pTheta[0];
                            sum += (float)alpha_k[1] * UnUPM[UnUPM[i].neighbors[j]].pTheta[1];
                            for (int k = 0; k < ClassNum; k++)
                                UnUPM[i].z_jk[j][k] = (float) ( alpha_k[k] * UnUPM[UnUPM[i].neighbors[j]].pTheta[k] ) /sum;
                        }
                    }//if(UnUPM[UnUPM[i].neighbors[j]].shill == -1 ){
                }//for (int j = 0; j < UnUPM[i].neighborsNum; j++){
            }//if (UnUPM[i].shill != -10){
        }//for (int i = 1; i <= unLen; i++){
    ComputeAlphaK(lambda, d);
}// End EStep_Crisp(float lambda, float d)

void IterLogReg(float lambda, float d){
    cout<<"Start Generating New Dataset...."<<endl;
    unsigned i,cl,k;
    int cc = 1; //New DataSet D', ID from 1!
    for (i = 1; i <= unLen; i++){
        if (UnUPM[i].shill != -10 ){
            if (UnUPM[i].shill == -1 && lambda > 0){ //Unlabeled User
                datasetForLR[cc].w_jk_i = lambda;
            }
            else if(UnUPM[i].shill == 0 || UnUPM[i].shill == 1){
                datasetForLR[cc].w_jk_i = 1;
            }
            datasetForLR[cc].tmpLab = UnUPM[i].tempLab;
            datasetForLR[cc].Data_Index = UnUPM[i].uID;
            cc++;
        }
    }//Insert Nodes Itselves
    cout<<"After Inserting Nodes ITSELVES, Data Size: "<<cc<<endl;
    int DD = d;
    for (i = 1; i <= unLen; i++){
        if (UnUPM[i].shill != -10 && UnUPM[i].neighborsNum > 0 && (UnUPM[i].pTheta[1]>=0.9 || UnUPM[i].pTheta[1]<=0.01)){
            for (cl = 0; cl < UnUPM[i].neighborsNum; cl++){
                if (UnUPM[UnUPM[i].neighbors[cl]].shill == -1){//NEED inserting two instances!!!
                    if (UnUPM[i].shill == 0 || UnUPM[i].shill == 1){
                        for(k = 0; k < ClassNum; k++){
                            datasetForLR[cc].w_jk_i = (double) DD * UnUPM[i].z_jk[cl][k] / UnUPM[i].neighborsNum;
                            datasetForLR[cc].tmpLab = k;
                            datasetForLR[cc].Data_Index = UnUPM[UnUPM[i].neighbors[cl]].uID;
                            cc++;
                        }
                    }//END if
                    else{
                        for(k = 0; k < ClassNum; k++){
                            datasetForLR[cc].w_jk_i = (double)lambda * DD * UnUPM[i].z_jk[cl][k] / UnUPM[i].neighborsNum;
                            datasetForLR[cc].tmpLab = k;
                            datasetForLR[cc].Data_Index = UnUPM[UnUPM[i].neighbors[cl]].uID;
                            cc++;
                        }
                    }//END else
                if(UnUPM[UnUPM[i].neighbors[cl]].shill == 0){
                    if (UnUPM[i].shill == 0 || UnUPM[i].shill == 1){
                        datasetForLR[cc].w_jk_i = (double) DD / UnUPM[i].neighborsNum;
                        datasetForLR[cc].tmpLab = 0;
                        datasetForLR[cc].Data_Index = UnUPM[UnUPM[i].neighbors[cl]].uID;
                        cc++;
                    }
                    else{
                        datasetForLR[cc].w_jk_i = (double)lambda * DD / UnUPM[i].neighborsNum;
                        datasetForLR[cc].tmpLab = 0;
                        datasetForLR[cc].Data_Index = UnUPM[UnUPM[i].neighbors[cl]].uID;
                        cc++;
                    }
                }//END if
                if(UnUPM[UnUPM[i].neighbors[cl]].shill == 1){
                    if (UnUPM[i].shill == 0 || UnUPM[i].shill == 1){
                        datasetForLR[cc].w_jk_i = (double) DD / UnUPM[i].neighborsNum;
                        datasetForLR[cc].tmpLab = 1;
                        datasetForLR[cc].Data_Index = UnUPM[UnUPM[i].neighbors[cl]].uID;
                        cc++;
                    }
                    else{
                        datasetForLR[cc].w_jk_i = (double)lambda * DD / UnUPM[i].neighborsNum;
                        datasetForLR[cc].tmpLab = 1;
                        datasetForLR[cc].Data_Index = UnUPM[UnUPM[i].neighbors[cl]].uID;
                        cc++;
                    }
                }
                }//End For cl
            }
        }//End if -10
    }//End for
    cout<<"Size of New DATASETS: "<<cc-1<<endl;
    int *tTrainIndex = new int[cc-1];
    int *tTrainLabel = new int[cc-1];
    float *tIWeight = new float[cc-1];
    for(i=0; i<cc-1; i++){
        tTrainIndex[i] = datasetForLR[i+1].Data_Index;
        tTrainLabel[i] = datasetForLR[i+1].tmpLab;
        tIWeight[i] = datasetForLR[i+1].w_jk_i;
    }
    Python_LR_Invoke(tTrainIndex,tTrainLabel,tIWeight,cc-1);
}//End IterLogReg(float lambda, float d)

int *Select_Top_Neighbor(int K, int index){
    int *top = new int[K];
    float currentMin = 1;
    int MinIndex = 1;
    int k = 0;
    unsigned j;
    for (j = 0; j < UnUPM[index].neighborsNum; j++){
        if(k<K){
            top[k]=j;
            k++;
        }
        else{
            for(unsigned p=0; p<K; p++){
                if(UnUPM[UnUPM[index].neighbors[top[p]]].pTheta[1]<currentMin){
                    currentMin = UnUPM[UnUPM[index].neighbors[top[p]]].pTheta[1];
                    MinIndex = p;
                }
            }
            if(UnUPM[UnUPM[index].neighbors[j]].pTheta[1]>currentMin){
                top[MinIndex] = j;
            }
        }
    }
    return top;
}

float MStep(float lambda, float d){
    unsigned i, j, k;
    float mul[ClassNum];
    float neighW[ClassNum];
    IterLogReg(lambda, d);
    for (i = 1; i <= unLen; i++) {
        if (UnUPM[i].shill == 1){
            UnUPM[i].pTheta[1] = 1;
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1];
        }
        else if (UnUPM[i].shill == 0){
            UnUPM[i].pTheta[1] = 0;
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1];
        }
        else if (UnUPM[i].shill == -1){
            UnUPM[i].pTheta[1] = Predit_Pro[i];
            UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1];
            if(UnUPM[i].neighborsNum>10 && UnUPM[i].pTheta[1]<0.4){//optional
                UnUPM[i].pTheta[1]+=0.2;
                UnUPM[i].pTheta[0] = 1 - UnUPM[i].pTheta[1];
            }
        }
    }// Finish, Update Probabilities
    
    for (i = 1; i <= unLen; i++) {
        if (UnUPM[i].shill == -1 && UnUPM[i].neighborsNum > 0){
            for (k = 0; k < ClassNum; k++) mul[k] = UnUPM[i].pTheta[k];
            if(mul[1]==0) mul[1]=0.0001;
            if(mul[0]==0) mul[0]=0.0001;
            for (k = 0; k < ClassNum; k++) neighW[k] = (float)1.0;
            for (j = 0; j < UnUPM[i].neighborsNum; j++){
                if(UnUPM[UnUPM[i].neighbors[j]].shill == -1){
                   float tempSum0 = 0;
                    tempSum0 += (float)(1- alpha_k[0]) * UnUPM[UnUPM[i].neighbors[j]].pTheta[0];
                    tempSum0 += (float)(1- alpha_k[1]) * UnUPM[UnUPM[i].neighbors[j]].pTheta[1];
                    tempSum0 = (float)pow(tempSum0, d/UnUPM[i].neighborsNum);
                    neighW[0] *= tempSum0;
                    float tempSum1 = 0;
                    tempSum1 += (float)alpha_k[0] * UnUPM[UnUPM[i].neighbors[j]].pTheta[0];
                    tempSum1 += (float)alpha_k[1] * UnUPM[UnUPM[i].neighbors[j]].pTheta[1];
                    tempSum1 = (float)pow(tempSum1, d/UnUPM[i].neighborsNum);
                    neighW[1] *= tempSum1;
                }
                if(UnUPM[UnUPM[i].neighbors[j]].shill == 0){
                    neighW[0] *= (float)pow(1- alpha_k[0], d/UnUPM[i].neighborsNum); //(1- alpha_k[0]);
                    neighW[1] *= (float)pow(alpha_k[0], d/UnUPM[i].neighborsNum); //alpha_k[0];
                }
                if(UnUPM[UnUPM[i].neighbors[j]].shill == 1){
                    neighW[1] *=(float)pow(alpha_k[1], d/UnUPM[i].neighborsNum);// alpha_k[1];
                    neighW[0] *= (float)pow(1- alpha_k[1], d/UnUPM[i].neighborsNum);//(1 -alpha_k[0]);
                }
            }//End for j
            for (k = 0; k < ClassNum; k++) mul[k] *= neighW[k];
            float sum = mul[0] + mul[1];
            mul[0] = (float)mul[0]/sum;
            mul[1] = (float)mul[1]/sum;
            for (k = 0; k < ClassNum; k++) {
                    UnUPM[i].pTheta[k] = mul[k];
                    UnUPM[i].pFinalLabelWeight[k] = mul[k];}
            if (mul[1] >= THRESHOLD) UnUPM[i].tempLab = 1;
            else UnUPM[i].tempLab = 0;
        }//END IF unlabeled
    }//End For i
    
    float Loss = LossFunction(lambda,d);
    int tp = 0; int fn = 0; int fp = 0; int tn = 0;
    ofstream fout("./spammer_results/prediction_5percents.txt");
    for (int i = 1; i <= unLen; i++) {
        if (UnUPM[i].shill == -1){
            if(UnUPM[i].tempLab == 1 && UnLabels[i] == 1) tp++;
            if(UnUPM[i].tempLab == 0 && UnLabels[i] == 1) fn++;
            if(UnUPM[i].tempLab == 1 && UnLabels[i] == 0) fp++;
            if(UnUPM[i].tempLab == 0 && UnLabels[i] == 0) tn++;
            fout << i << " " << UnUPM[i].tempLab << endl;
        }
    }
    fout.close();
    cout<<"tp = "<<tp<<", Spammer to Spammer"<<endl;
    cout<<"fn = "<<fn<<", Spammer to Normal"<<endl;
    cout<<"fp = "<<fp<<", Normal to Spammer"<<endl;
    cout<<"tn = "<<tn<<", Normal to Normal"<<endl;

    float recall = (float)tp/(tp+fn);
    float precision = (float)tp/(tp+fp);
    float f = 2*recall*precision /(recall + precision);

    cout<<"RECALL = "<<recall<<endl;
    cout<<"PRECISION = "<<precision<<endl;
    cout<<"F-MEASURE = "<<f<<endl;
    return Loss;
}//END void MStep(float lambda, float d)


void Output_to_File(int k){
    string file = "./our_output_files/AmazonLRF";
    file.append(to_string(k));
    ofstream fout2(file);
    for(int i = 1; i <= unLen; i++){
            if (UnUPM[i].shill == -1){
                fout2<<i<<","<<UnUPM[i].pTheta[1]<<endl;
            }
        }
}

/*——————————Overall Procedure of Algorithm——————————————————————————————————————————*/
void control(float lambda, float d){
    initialization(); 
    cout<<"The lambda is: "<<lambda<<endl;
    InitClassifier(lambda);
    int IterNum = 0;
    float Loss;
    while (IterNum < MAXITTOTAL){
        EStep_Crisp((float)lambda,(float)d);
        Loss=MStep((float)lambda,(float)d);
        IterNum++;
        cout<<"Iteration Number: "<<IterNum<<", Loss: "<<Loss<<endl;
        Output_to_File(IterNum);
    }
}//End void control(float lambda, float d)

int main(int argc, const char * argv[]) {
    control(LL, DD);
    return 0;
}
