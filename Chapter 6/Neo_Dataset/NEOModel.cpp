// NEOModel.cpp
#include "iostream"
#include "NeuralNetwork.h"
using namespace ArtificialNeuralNetwork;
using namespace std;
int main(){
    // Creating the neural network model
    NeuralNetwork Model;
    arma::field<dmat> X_Train;
    arma::field<dmat> Y_Train;
    arma::field<dmat> X_Test;
    arma::field<dmat> Y_Test;
    X_Train.load("X_TRAIN_NEO_FIELD.bin");
    Y_Train.load("Y_TRAIN_NEO_FIELD.bin");
    X_Test.load("X_TEST_NEO_FIELD.bin");
    Y_Test.load("Y_TEST_NEO_FIELD.bin");

    Model.Add(Model.InputLayer(5));
    Model.Add(Model.DenseLayer(20,NeuralNetwork::Activations::Relu));
    Model.Add(Model.DenseLayer(10,NeuralNetwork::Activations::Relu));
    Model.Add(Model.DenseLayer(5,NeuralNetwork::Activations::Relu));
    Model.Add(Model.DenseLayer(1,NeuralNetwork::Activations::Sigmoid));
    Model.Summary();

    OptimizerClass Optimizer;
    Optimizer.RMSProp(0.01,0.9,1e-7);  

    map<string,dmat> TRAIN_OUTPUTS = Model.Train(
        X_Train,
        Y_Train,
        Optimizer,
        100,
        400,
        NeuralNetwork::Losses::BinaryCrossEntropy);
   
    bool lossSave = TRAIN_OUTPUTS["losses"].save("NEO_losses.csv", arma::csv_ascii);
    if(lossSave)cout<<"Lossess saved sucessfull."<<endl;

    arma::field<dmat> Predictions;
    Predictions=Model.Predict(X_Test);
    dmat OUTPUTS(Predictions.size(),1);
    for(int i=0;i<Predictions.size();i++){
        if(Predictions(i)[0]<0.5){
            OUTPUTS[i]=0;
        }
        else{
            OUTPUTS[i]=1;
        }
    }
    bool save = OUTPUTS.save("Y_PREDICTS.csv",arma::csv_ascii);
    if(save)cout<<"Predictions generation sucessfull";
    return 0;
}
