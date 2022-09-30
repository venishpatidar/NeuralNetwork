// PlannerDataset.cpp
#include "iostream"
#include "NeuralNetwork.h"
using namespace ArtificialNeuralNetwork;
using namespace std;
int main(){
    // Creating the neural network model
    NeuralNetwork Model;
    int _dataset;
    cout<<"Training on dataset\n";
    cout<<"0. Circles\n1. Flowers\n2. Gaus\n3. Moons\n";
    cin>>_dataset;
    string Dataset;
    switch (_dataset) {
        case 0:
            Dataset = "CIRCLES";
            break;
        case 1:
            Dataset = "FLOWERS";
            break;    
        case 2:
            Dataset = "GAUS";
            break;
        case 3:
            Dataset = "MOONS";
            break;
        default:
            Dataset = "CIRCLES";
            break;
    }

    arma::field<dmat> X_Train;
    arma::field<dmat> Y_Train;
    arma::field<dmat> X_Test;
    X_Train.load(Dataset+"/X_TRAIN_"+Dataset+"_FIELD.bin");
    Y_Train.load(Dataset+"/Y_TRAIN_"+Dataset+"_FIELD.bin");
    X_Test.load(Dataset+"/X_TEST_"+Dataset+"_FIELD.bin");

    Model.Add(Model.InputLayer(2));
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
        1000,
        400,
        NeuralNetwork::Losses::BinaryCrossEntropy);
   
    bool lossSave = TRAIN_OUTPUTS["losses"].save(Dataset+"/"+Dataset+"_losses.csv", arma::csv_ascii);
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
    bool save = OUTPUTS.save(Dataset+"/Y_PREDICTS.csv",arma::csv_ascii);
    if(save)cout<<"Predictions generation sucessfull";
    return 0;
}
