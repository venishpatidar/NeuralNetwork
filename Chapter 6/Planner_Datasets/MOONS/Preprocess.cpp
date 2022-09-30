/**
 * @file Preprocess.cpp
 * @author Venish Patidar
 * @brief This File converts train and test csv files of planner datasetpoints to optimized 
 *        armadillo field file in order to use it in training. It will change following files
 *        @param X_TRAIN_ @ref PlannerDatasetGenerator.ipynb 
 *        @param Y_TRAIN_ @ref PlannerDatasetGenerator.ipynb 
 *        @param X_TEST_ @ref PlannerDatasetGenerator.ipynb 
 *        @param Y_TEST_ @ref PlannerDatasetGenerator.ipynb 
 *        into:
 *          X_TRAIN_.csv -> @return X_TRAIN_field.bin
 *          Y_TRAIN_.csv -> @return Y_TRAIN_field.bin
 * 
 *          X_TEST_.csv -> @return X_TEST_field.bin
 *          Y_TEST_.csv -> @return Y_TEST_field.bin
 * 
 * @version 0.1
 * @date 2022-08-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */



#include <stdio.h>
#include "iostream"
#include <stdlib.h>
#include <armadillo>
#include <math.h>
#include "iomanip"
#include <fstream>
using namespace arma;
using namespace std;


int n_train = 400;
int n_test =  252070; // number of points in grid @ref PlannerDatasetGenerator.ipynb Generating the decision boundary's grid of points
int input_dim = 2;
int output_dim = 1;

string X_TRAIN = "X_TRAIN_MOONS";
string Y_TRAIN = "Y_TRAIN_MOONS";
string X_TEST = "X_TEST_MOONS";



void PROCESS_X_TRAIN(int number_of_samples,int number_of_elements_in_row_X,string inputfilename,string outputfilename){
    dmat Samples;
    Samples.load(inputfilename+".csv");
    arma::field<dmat> X_train(number_of_samples,1);
    for(int i=0;i<number_of_samples;i++){
        dmat Temp(number_of_elements_in_row_X,1);
        for(int j=0;j<number_of_elements_in_row_X;j++){
            Temp[j]=Samples.row(i)[j];
        }
        
        X_train[i]=Temp;
    }
    bool operation = X_train.save(outputfilename+".bin");
    if(operation){
        cout<<"saving "<<outputfilename<<".bin is successfull\n";
    }
}

void PROCESS_X_TEST(int number_of_samples,int number_of_elements_in_row_X,string inputfilename,string outputfilename){
    dmat Samples;
    Samples.load(inputfilename+".csv");
    arma::field<dmat> X_test(number_of_samples,1);
    for(int i=0;i<number_of_samples;i++){
        dmat Temp(number_of_elements_in_row_X,1);
        for(int j=0;j<number_of_elements_in_row_X;j++){
            Temp[j]=Samples.row(i)[j];
        }
        X_test[i]=Temp;
    }
    bool operation = X_test.save(outputfilename+".bin");
    if(operation){
        cout<<"saving "<<outputfilename<<".bin is successfull\n";
    }
}

void PROCESS_Y_TRAIN(int number_of_samples,int number_of_elements_in_row_Y,string inputfilename,string outputfilename){
    dmat OutputSamples;
    OutputSamples.load(inputfilename+".csv");
    arma::field<dmat> Y_train(number_of_samples,1);
    for(int i=0;i<number_of_samples;i++){
        dmat Temp(number_of_elements_in_row_Y,1);
        for(int j=0;j<number_of_elements_in_row_Y;j++){
            Temp[j]=OutputSamples.row(i)[j];
        }
        Y_train[i]=Temp;
    }
    bool operation = Y_train.save(outputfilename+".bin");
    if(operation){
        cout<<"saving "<<outputfilename<<".bin is successfull\n";
    }
}



int main(){

    int number_of_samples_train=n_train;
    int number_of_samples_test=n_test;
    int number_of_elements_in_row_X=input_dim;
    int number_of_elements_in_row_Y=output_dim;

    PROCESS_X_TRAIN(number_of_samples_train,number_of_elements_in_row_X,X_TRAIN,X_TRAIN+"_FIELD");

    PROCESS_Y_TRAIN(number_of_samples_train,number_of_elements_in_row_Y,Y_TRAIN,Y_TRAIN+"_FIELD");

    PROCESS_X_TEST(number_of_samples_test,number_of_elements_in_row_X,X_TEST,X_TEST+"_FIELD");
    

    return 0;
}

