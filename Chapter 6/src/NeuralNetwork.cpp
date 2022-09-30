// NeuralNetwork.cpp

#include "NeuralNetwork.h"
#include <bits/stdc++.h>
#include "iostream"
#include <map>
using namespace std;

namespace ArtificialNeuralNetwork{

// Activation class members  
    dmat ActivationClass::CalculateActivationFunction(dmat matrix,ActivationFunction activation){
        switch (activation){
            case ActivationClass::ActivationFunction::Linear:
                return ActivationClass::LinearFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::BinaryStep:
                return ActivationClass::BinaryStepFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::Relu:
                return ActivationClass::ReluFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::LeakyReLU:
                return ActivationClass::LeakyReLUFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::Sigmoid:
                return ActivationClass::SigmoidFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::Tanh:
                return ActivationClass::TanhFunction(matrix);
                break;
            default:
                return ActivationClass::ReluFunction(matrix);
                break;
        };
        return ActivationClass::ReluFunction(matrix);
    };

    dmat ActivationClass::LinearFunction(dmat matrix){
        return matrix;
    };
    dmat ActivationClass::BinaryStepFunction(dmat matrix){
        // In this transform it is one line if statement which will
        // iterate through all values inside the matrix and transform the
        // values accordingly
        return matrix.transform( [](double val) { if(val<0.5)return 0.0;else return 1.0; } );
    };
    dmat ActivationClass::ReluFunction(dmat matrix){
        return matrix.transform( [](double val) { if(val<0)return 0.0;else return val; } );
    };
    dmat ActivationClass::LeakyReLUFunction(dmat matrix){
        double alpha = 0.01;
        return matrix.transform( [alpha](double val) { if(val<0) return (alpha*val);else return val; } );
    };
    dmat ActivationClass::SigmoidFunction(dmat matrix){
        dmat temp;
        //this would be calculated elementwise
        temp = (1/(1+exp(-matrix)));
        return temp;
    };
    dmat ActivationClass::TanhFunction(dmat matrix){
        //this would be calculated elementwise
        return tanh(matrix);
    };


    dmat ActivationClass::CalculateDerivativeActivationFunction(dmat matrix,ActivationFunction activation){
        switch (activation){
            case ActivationClass::ActivationFunction::Linear:
                return ActivationClass::DerivativeLinearFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::BinaryStep:
                return ActivationClass::DerivativeBinaryStepFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::Relu:
                return ActivationClass::DerivativeReluFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::LeakyReLU:
                return ActivationClass::DerivativeLeakyReluFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::Sigmoid:
                return ActivationClass::DerivativeSigmoidFunction(matrix);
                break;
            case ActivationClass::ActivationFunction::Tanh:
                return ActivationClass::DerivativeTanhFunction(matrix);
                break;
            default:
                return ActivationClass::DerivativeReluFunction(matrix);
                break;
        };
        return ActivationClass::DerivativeReluFunction(matrix);
    };
    dmat ActivationClass::DerivativeLinearFunction(dmat matrix){
        dmat temp=matrix;
        temp.fill(arma::fill::ones);
        return temp;
    };
    dmat ActivationClass::DerivativeBinaryStepFunction(dmat matrix){
        dmat temp=matrix;
        temp.fill(arma::fill::zeros);
        return temp;
    };
    dmat ActivationClass::DerivativeReluFunction(dmat matrix){
        return matrix.transform( [](double val) { if(val<0)return (double)0.0;else return (double)1; } );
    };
    dmat ActivationClass::DerivativeLeakyReluFunction(dmat matrix){
        double alpha = 0.01;
        return matrix.transform( [alpha](double val) { if(val<0)return alpha;else return (double)1; } );
    };
    dmat ActivationClass::DerivativeSigmoidFunction(dmat matrix){
        return (SigmoidFunction(matrix)%(1-SigmoidFunction(matrix)));
    };
    dmat ActivationClass::DerivativeTanhFunction(dmat matrix){
        return (1-(TanhFunction(matrix)%TanhFunction(matrix)));
    };


// Losses class methods

    double LossesClass::CalculateLoss(dmat y_true,dmat y_pred,LossFunctions loss){
        switch (loss) {
            case LossFunctions::MeanSquareError:
                return MeanSquareErrorFunction(y_true,y_pred);
                break;
            case LossFunctions::MeanAbsoluteError:
                return MeanAbsoluteErrorFunction(y_true,y_pred);
                break;
            case LossFunctions::BinaryCrossEntropy:
                return BinaryCrossEntropyFunction(y_true,y_pred);
                break;
            case LossFunctions::CategoricalCrossEntropy:
                return CategoricalCrossEntropyFunction(y_true,y_pred);
                break;
            default:
                break;
        }
        return BinaryCrossEntropyFunction(y_true,y_pred);
    }

    dmat LossesClass::CalculateDerivativeLoss(dmat y_true,dmat y_pred,LossFunctions loss){
        switch (loss) {
            case LossFunctions::MeanSquareError:
                return DerivativeMeanSquareErrorFunction(y_true,y_pred);
                break;
            case LossFunctions::MeanAbsoluteError:
                return DerivativeMeanAbsoluteErrorFunction(y_true,y_pred);
                break;
            case LossFunctions::BinaryCrossEntropy:
                return DerivativeBinaryCrossEntropyFunction(y_true,y_pred);
                break;
            case LossFunctions::CategoricalCrossEntropy:
                return DerivativeCategoricalCrossEntropyFunction(y_true,y_pred);
                break;
            default:
                break;
        }
        return DerivativeBinaryCrossEntropyFunction(y_true,y_pred);
    }


    double LossesClass::MeanSquareErrorFunction(dmat y_true,dmat y_pred){
        double MSE = ((double)1/(double)y_true.n_elem)*arma::accu(arma::pow((y_pred-y_true),2));
        return MSE;
    };
    double LossesClass::MeanAbsoluteErrorFunction(dmat y_true,dmat y_pred){
        double MAE = ((double)1/(double)y_true.n_elem)*arma::accu(arma::abs(y_pred-y_true));
        return MAE;
    };
    double LossesClass::BinaryCrossEntropyFunction(dmat y_true,dmat y_pred){
        double Epsilon = 1e-7;
        y_pred = y_pred.clamp(Epsilon,1 - Epsilon);
        double BCE = -((double)1/(double)y_true.n_elem)*arma::accu( ((y_true)%(log(y_pred+Epsilon)) ) + ( (1-y_true)%(log(1-y_pred+Epsilon)) ) );
        return BCE;
    };
    double LossesClass::CategoricalCrossEntropyFunction(dmat y_true,dmat y_pred){
        double Epsilon = 1e-7;
        y_pred = y_pred.clamp(Epsilon,1 - Epsilon);
        double CCE = arma::accu(-(y_true)%(log(y_pred+Epsilon)));
        return CCE;
    };

    dmat LossesClass::DerivativeMeanSquareErrorFunction(dmat y_true,dmat y_pred){
        dmat DMSE = ((double)2/(double)y_true.n_elem)*(y_pred-y_true);
        return DMSE;
    }

    dmat LossesClass::DerivativeMeanAbsoluteErrorFunction(dmat y_true,dmat y_pred){
        dmat DMSE = ((double)2/(double)y_true.n_elem)*(y_pred-y_true);
        return DMSE;
    }
    dmat LossesClass::DerivativeBinaryCrossEntropyFunction(dmat y_true,dmat y_pred){
        dmat DMSE = ((double)2/(double)y_true.n_elem)*(y_pred-y_true);
        return DMSE;
    }
    dmat LossesClass::DerivativeCategoricalCrossEntropyFunction(dmat y_true,dmat y_pred){
        dmat DMSE = ((double)2/(double)y_true.n_elem)*(y_pred-y_true);
        return DMSE;
    }

// Optimizer class members  
    OptimizerClass::OptimizerClass(){
        learning_rate=0.001;
        epsilon=1e-7;
        rho=0.9;
        SelectedOptimizer = OptimizersEnum::GradientDescentEnum;
    }
    /**
     * @brief GradientDescent
     * Set the optimizer to use Adaptive Gradient Descent
     * and use the passed hyperparameters
     *  @param LearningRate The learning rate of gradient descent 
     */
    void OptimizerClass::GradientDescent(double LearningRate=0.001){
        learning_rate = LearningRate;
        SelectedOptimizer=OptimizersEnum::GradientDescentEnum;
    }
    /**
     * @brief AdaGrad
     * Set the optimizer to use Adaptive Gradient Descent
     * and use the passed hyperparameters
     * @param LearningRate The inital learning rate
     * @param Epsilon The small value to keep the denom positive
     */
    void OptimizerClass::AdaGrad(double LearningRate=0.001,double Epsilon=1e-7){
        learning_rate = LearningRate;
        epsilon = Epsilon;
        SelectedOptimizer=OptimizersEnum::AdaGradEnum;
    }
    /**
     * @brief RMSProp
     * Set the optimizer to use RMSProp
     * and use the passed hyperparameters 
     * 
     * @param LearningRate The inital learning rate
     * @param Rho The hyperparameter rho
     * @param Epsilon The small value to keep the denom positive
     */
    void OptimizerClass::RMSProp(double LearningRate=0.001,double Rho=0.9,double Epsilon=1e-7){
        learning_rate = LearningRate;
        epsilon = Epsilon;
        rho = Rho;
        SelectedOptimizer=OptimizersEnum::RMSPropEnum;
    }

    
    dmat OptimizerClass::CalculateSelectedOptimizer(dmat derivatives, string ref_name){
        switch (SelectedOptimizer){
        case OptimizersEnum::GradientDescentEnum:
            return CalculateGradientDescent(derivatives);
            break;

        case OptimizersEnum::AdaGradEnum:
            return CalculateAdaGrad(derivatives);
            break;
        case OptimizersEnum::RMSPropEnum:
            return CalculateRMSProp(derivatives,ref_name);
            break;
        default:
            return CalculateGradientDescent(derivatives);
            break;
        }
        return CalculateGradientDescent(derivatives);
    }

    dmat OptimizerClass::CalculateGradientDescent(dmat derivatives){
        /*
            For Gradient Descent, the updating term is calculated as
            delta_W = learning_rate*derivatives
        */
        return learning_rate*derivatives;
    }
    dmat OptimizerClass::CalculateAdaGrad(dmat derivatives){
        /*
            For Adaptive Gradient Descent, the updating term is calculated as
            delta_W = (learning_rate/(root(G_old+epsilon)))*derivatives
        */
        double GradientSquare = arma::accu(arma::pow(derivatives,2));
        double Denominator = sqrt(GradientSquare+epsilon);
        return (learning_rate/Denominator)*(derivatives);
    }
    dmat OptimizerClass::CalculateRMSProp(dmat derivatives,string ref_name){
        /*
            For RMSProp, the updating term is calculated as
            V = rho(prev state V) + (1-rho)8 Square of derivatives or gradient
            delta_W = (learning_rate/(root(V+epsilon)))*derivatives
        */
        dmat prev_state_v;
        if(previous_state[ref_name].size()==0){
            prev_state_v=derivatives;
            prev_state_v.fill(arma::fill::zeros);
        }
        else{
            prev_state_v=previous_state[ref_name];
        }
        dmat DerivativeSquare = arma::pow(derivatives,2);
        dmat V = rho*prev_state_v + (1-rho)*DerivativeSquare;
        previous_state[ref_name] = V;
        return (learning_rate/(arma::sqrt(V+epsilon)))%derivatives;
    }



    /**
     * @brief Optimize
     * This function takes the derivative and the parameter of
     * the network and perform the optimization, update the 
     * parameters and returns the updated map of parameters.
     * 
     * @param derivatives The map of calculated derivatives of the network summed over a batch of input
     * @param parameters The map of parameters of each layer's weights and biases
     * @param batch_size The size of a batch of input used to average the derivatives 
     * @param total_layers The total layer of the network used to loop over it.
     * @return map<string,dmat> 
     */
    map<string,dmat> OptimizerClass::Optimize(map<string,dmat> derivatives,map<string,dmat> parameters,int batch_size,int total_layers){
        // looping over the layers
        for(int layer_number=1;layer_number<=total_layers;layer_number++){
            // averaging the gradient sum over the batch of input
            dmat batch_average_d_weights = ((double)1.0/(double)batch_size)*derivatives["dW"+to_string(layer_number)];
            dmat batch_average_d_bias = ((double)1.0/(double)batch_size)*derivatives["dB"+to_string(layer_number)];

            // calculating the updating term 
            dmat delta_W_grads = CalculateSelectedOptimizer(
                    batch_average_d_weights,
                    "dW"+to_string(layer_number)
                );

            dmat delta_B_grads = CalculateSelectedOptimizer(
                    batch_average_d_bias,
                    "dB"+to_string(layer_number)
                );

            // Updating the parameters
            parameters["W"+to_string(layer_number)]-= delta_W_grads;
            parameters["B"+to_string(layer_number)]-= delta_B_grads;
        }

        // returning the updated parameter
        return parameters;
    }





// Neural Network members

    NeuralNetwork::NeuralNetwork(){
        arma::arma_rng::set_seed_random(); 
        RecordIndex=0;
        Records=(LayerBP *)malloc(sizeof(LayerBP));
        InputLayerFlag=false;
    }

    /**
     * ************
     * @brief InputLayer 
     * ************ 
     * This function is used to define the input layer to 
     * the network.
     * @param Neurons number of input neurons.
     * @returns information of the layer in LayerBP struct form.
    */
    NeuralNetwork::Layer NeuralNetwork::InputLayer(int Neurons){
        
        if(InputLayerFlag){
            cout<<"Input layer had already been defined.\n";
            throw;
        }
        else{
            LayerBP newLayer;
            newLayer.LayerNumber=RecordIndex;
            newLayer.Neurons=Neurons;
            newLayer.LayerType=INPUT;
            InputLayerFlag=true;
            return newLayer;
        }
    };

    /**
     * ************
     * @brief DenseLayer 
     * ************ 
     * This function is used to define the dense layer of the function
     * 
     * @param Neurons number of neurons of the current layer.
     * @param ActivationFunction activation function of neurons.
     * @returns information of the layer in LayerBP struct form.
    */
    NeuralNetwork::Layer NeuralNetwork::DenseLayer(int Neurons, Activations ActivationFunction){
        
       if(!InputLayerFlag){
            cout<<"Input layer had not been defined yet.\n";
            throw;
        }
        else{
            LayerBP newLayer;
            newLayer.LayerNumber=RecordIndex;
            newLayer.Activation=ActivationFunction;
            newLayer.Neurons=Neurons;
            newLayer.LayerType=DENSE;
            return newLayer;
        }
    };

    /**
     * ************
     * @brief Add
     * ************ 
     * 1. This function adds a new layer that is passed to it to Records, 
     *    increments RecordIndex and reallocates the size of Records and 
     *    increases it by one unit size.
     * 2. This function also initializes parameters for a particular layer 
     *    and store it in the parameters dictionary at its appropriate 
     *    position.
     * 
     * @param layer Layer map which needs to be added to the neural network.
    */
    void NeuralNetwork::Add(Layer layer){

        /*
            pos of any hidden layer is W(a) or B(a) where a is the number of  
            layers deep where it is located.
        */
        int pos=RecordIndex;
        int current_layer_neurons = layer.Neurons;
        int prev_layer_neurons=-1;
        if(RecordIndex>0) prev_layer_neurons=Records[RecordIndex-1].Neurons;
        /* 
            Trainable parameters are weights and biases of the current layer
            which can be calculated by previous input neurons times current   
            neurons which result in total weights parameters and total bias 
            parameters that are the number of neurons of the current layer  
            hence total trainable parameters are 
            (prev_neurons*current_neurons)+current_neurons
            or 
            (current_layer_neurons)*(prev_layer_neurons+1)
        */
        layer.TotalTrainableParameters = (current_layer_neurons)*(prev_layer_neurons+1);
        
        // Initialize weights and biases parameters 
        initialize_paramaeters(
            layer.Neurons, 
            prev_layer_neurons,
            pos
        );

        // Add to records array
        add_to_record(layer);
    };



    /**
     * ********************************
     *  @brief initialize_paramaeters  
     * ********************************
     * It initializes the parameters as of dimension supplied to it 
     * and then stores these randomly generated weights and biases
     * at location W+pos for weights and B+pos for biases in
     * parameters map.
     * 
     * In case of input layer that is at position 0 set
     * Create an input signal dimension zero array matrix.
     * 
     * @param n_x x dimension i.e neurons of current layer.
     * @param n_y y dimension i.e neurons of the previous layer.
     * @param pos int which will be used to store at that position
     * 
    */
    void NeuralNetwork::initialize_paramaeters(int n_x, int n_y, int pos){
        if(pos==0){
            dmat X(n_x,1);
            X.fill(0.0);
            parameters["X"+to_string(pos)] = X;
        }
        else{
            dmat W(n_x,n_y,arma::fill::randn);
            dmat B(n_x,1,arma::fill::randn);
            parameters["W"+to_string(pos)] = W;
            parameters["B"+to_string(pos)] = B;
        }
    }


    /**
     * **********************
     * @brief add to record 
     * **********************
     * 1. This function adds the supplied layer's information
     *    to Record array. 
     * 2. Realloactes the Records array to increase the size to 
     *    one LayerBP unit size.
     * @param layer Record of the corresponding record to add.
    */
    void NeuralNetwork::add_to_record(LayerBP layer){
        // Stores the information of layer to Records array
        Records[RecordIndex]=layer;
        // Dynamically increases the array size
        Records = (LayerBP *)realloc(Records, sizeof(LayerBP)*(RecordIndex+2));
        RecordIndex++;
    };



    /**
     * **********************
     * @brief PrintMap 
     * **********************
     * This function prints all the weights and biases 
     * matrixes with their values and name
    */
    void NeuralNetwork::PrintMap(){
        map<string,dmat> m = parameters;
        cout << "[ \n";
        for (auto &item : m) {
            cout << item.first << ":" << " ";
            item.second.brief_print();
        }
        cout << "]\n";
    };


    /**
     * **********************
     * @brief Summary 
     * **********************
     * Prints the systematic architecture 
     * of network
    */
    void NeuralNetwork::Summary(){
        auto DrawHorizontalLine = [](char fill='-',int width=20*5) {cout<<setfill(fill)<<setw(width);cout<<""<<endl;cout<<setfill(' ');};
        DrawHorizontalLine();
        cout<<left<<setw(2)<<"|"<< left<< setw(36) << "Layer(Type)_Layer(Number)"<<right<<setw(2)<<"|"
            << right<< setw(18) << "Neurons"<<right<<setw(2)<<"|"
            << right<< setw(18) << "Activation"<<right<<setw(2)<<"|"
            << right<< setw(18) << "Parameters"<<right<<setw(2)<<"|"
            << endl;
        DrawHorizontalLine('=');
        for(int i=0;i<RecordIndex;i++){
            string layerType;
            string activationType;
            if(Records[i].LayerType== INPUT) layerType="INPUT";
            else layerType="DENSE";
            if(i==0) activationType="-";
            else if (Records[i].Activation==Activations::Linear) activationType="Linear";
            else if (Records[i].Activation==Activations::BinaryStep) activationType="BinaryStep";
            else if (Records[i].Activation==Activations::Relu) activationType="Relu";
            else if (Records[i].Activation==Activations::LeakyReLU) activationType="LeakyReLU";
            else if (Records[i].Activation==Activations::Sigmoid) activationType="Sigmoid";
            else if (Records[i].Activation==Activations::Tanh) activationType="Tanh";

            cout<<left<<setw(2)<<"|"<< left<< setw(36) <<layerType+"_"+to_string(i)<<right<<setw(2)<<"|"
                << right<< setw(18) << Records[i].Neurons<<right<<setw(2)<<"|"
                << right<< setw(18) << activationType<<right<<setw(2)<<"|"
                << right<< setw(18) << Records[i].TotalTrainableParameters<<right<<setw(2)<<"|"
                << endl;
            DrawHorizontalLine();
        }
    };



    /**
     * **********************
     * @brief Feedforward
     * **********************
     * Runs a forward pass of the network-based
     * on the input that has passed and returns 
     * the tuple of output that is the value of last 
     * layer, and the map of values of all the 
     * layer.
     * 
     * @param input The input matrix to the network.
     * @return tuple<dmat, map<string,dmat>> 
     */
    tuple<dmat, map<string,dmat>> NeuralNetwork::Feedforward(dmat input){

        map<string,dmat> cache;
        dmat output;
        cache["X0"] = input;


        for(int layer_number=1;layer_number<RecordIndex;layer_number++){

            dmat Calculation;
            Calculation = parameters["W"+to_string(layer_number)]*cache["X"+to_string(layer_number-1)];
            Calculation = Calculation+parameters["B"+to_string(layer_number)];
            
            //calculating the activation function                    
            Calculation = ActivationClass::CalculateActivationFunction(
                            Calculation,
                            Records[layer_number].Activation
                        );
            cache["X"+to_string(layer_number)] = Calculation;
        }
            
 
        output = cache["X"+to_string(Records[RecordIndex-1].LayerNumber)];
        return {output,cache};
    };



    /**
     * **********************
     * @brief Backpropagation
     * **********************
     * The backpropagation function generates the 
     * derivative map of the network. It returns the
     * sum of derivatives over a batch. The output still 
     * needed to be averaged after the last input of the batch.
     * The average that is dividing by the batch size happens 
     * in the optimizer.
     * 
     * @param DerivativeLoss The derivative of the loss of network
     * @param cache The map of each layer's output
     * @param derivatives The map where previous inputs of the batch's derivative are stored
     * @return map<string,dmat> 
     */

    map<string,dmat> NeuralNetwork::Backpropagation(dmat DerivativeLoss, map<string,dmat> cache, map<string,dmat> derivatives){
        int Output_layer_number = Records[RecordIndex-1].LayerNumber;
        dmat y_pred = cache["X"+to_string(Output_layer_number)];

        dmat output_layer_activation_derivative=ActivationClass::CalculateDerivativeActivationFunction(y_pred,Records[RecordIndex-1].Activation);
        derivatives["dt"+to_string(Output_layer_number)] = DerivativeLoss%output_layer_activation_derivative;        

        if(derivatives["dW"+to_string(Output_layer_number)].size()==0){
            /* First input of the batch hence 
             Initiate the first and then start 
             adding for rest inputs of the batch*/
            derivatives["dW"+to_string(Output_layer_number)] =  derivatives["dt"+to_string(Output_layer_number)]*cache["X"+to_string(Output_layer_number-1)].t();
        }
        else{
            derivatives["dW"+to_string(Output_layer_number)] +=  derivatives["dt"+to_string(Output_layer_number)]*cache["X"+to_string(Output_layer_number-1)].t();
        }
        if(derivatives["dB"+to_string(Output_layer_number)].size()==0){
            derivatives["dB"+to_string(Output_layer_number)] =  derivatives["dt"+to_string(Output_layer_number)];
        }
        else{
            derivatives["dB"+to_string(Output_layer_number)] +=  derivatives["dt"+to_string(Output_layer_number)];
        }

        
        for(int layer_number=RecordIndex-2;layer_number>=1;layer_number--){
            // Calculating the f'(Z)
            dmat ActivationFunctionDerivative = ActivationClass::CalculateDerivativeActivationFunction(
                cache["X"+to_string(layer_number)],
                Records[layer_number].Activation
            );
            // Calculating the variable delta for layer = layer number
            // delta_x = [(W_x+1)T * delta_x+1] % f'(Z) 
            derivatives["dt"+to_string(layer_number)] = parameters["W"+to_string(layer_number+1)].t() * derivatives["dt"+to_string(layer_number+1)]; 

            // Calculating the weights and bias variable for a batch
            // dW_x = delta_x * (X_x-1)T and
            // dB_x = dela_x
            if(derivatives["dW"+to_string(layer_number)].size()==0){
                derivatives["dW"+to_string(layer_number)] =  derivatives["dt"+to_string(layer_number)]*cache["X"+to_string(layer_number-1)].t();
            }
            else{
                derivatives["dW"+to_string(layer_number)] +=  derivatives["dt"+to_string(layer_number)]*cache["X"+to_string(layer_number-1)].t();
            }
            if(derivatives["dB"+to_string(layer_number)].size()==0){
                derivatives["dB"+to_string(layer_number)] =  derivatives["dt"+to_string(layer_number)];
            }
            else{
                derivatives["dB"+to_string(layer_number)] +=  derivatives["dt"+to_string(layer_number)];
            }
        }
        return derivatives;
    };
    /**
     * @brief Prints out the current progress of 
     *  batches with loading bar
     * 
     * @param batch_number current batch number 
     * @param total_batches total batches
     * @param avg_batch_loss average batch loss
     * @param barWidth width of the loading bar
     */
    void NeuralNetwork::print_progress_bar(int batch_number, int total_batches,double avg_batch_loss, int barWidth=50){
        /*
            For each epoch, progress is at what batch currently 
            the loop is. For batch one, progress would be zero 
            divide by batch size, which is zero percent.
        */
        float progress =  (float)(batch_number+1)/(float)total_batches;
        cout << "[";
        int pos = barWidth * progress;
        // Printing the progress
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] " <<(batch_number+1)<<"/"<<total_batches<<"\t losses: "<<avg_batch_loss <<"\r";
        cout.flush();
    };

    /**
     * @brief This functions trains the 
     * neural network, minimizes the losses 
     * and increase the accuracies of predicton
     * 
     * @param input list of dmat of inputs
     * @param output list of dmat of actual outputs
     * @param optimizer optimizer to optimize the network
     * @param epochs number of iterations
     * @param batch_size size of one batch of input
     * @param loss loss function of the network
     * @return map<string,dmat> it returns losses
     */

    map<string,dmat> NeuralNetwork::Train(arma::field<dmat> input, arma::field<dmat> output,OptimizerClass optimizer,int epochs, int batch_size=1,NeuralNetwork::Losses loss=NeuralNetwork::Losses::BinaryCrossEntropy){
        
        // Calculating the batch size 
        cout<<endl;
        int input_size=input.size();
        cout<<"Input Size "<<input_size<<endl;
        if(input_size<batch_size)throw;
        cout<<"Batches would be of "<<batch_size<<endl;
        int total_batches = ceil(((double)input_size/(double)batch_size));
        cout<<"Total batches are "<<total_batches<<endl;

        // Initializing the batch variables
        int current=0;
        int end=batch_size;
        int batch_number=0;

        // losses matrix to store the loss of each epoch
        dmat losses(epochs,1);
        map<string,dmat> RETURN_MAP;
            

        /*
            Loop over epochs from 0 to epochs
        */
        for(int epoch=0;epoch<epochs;epoch++){
            cout<<"Epoch "<<epoch<<endl;

            /*Preparing thee initializers for batch formations */
            int current=0;      /*Setting starting point of batch to zero*/
            int end=batch_size; /*setting ending point of batch to current batch size*/
            int batch_number=0; /*setting batch number to zero to track progress later*/
            double EPOCH_LOSS=0.0; /*Initializing the epoch loss*/

            while(current<input_size){
                /*
                    Initializer for tracking 
                    the number of inputs in a batch
                */
                int j=0;
                /*
                    Initializing and resetting 
                    the batch loss to zero
                */
                int BATCH_LOSS=0; 
                /*
                    Initializing the derivatives
                    map for a batch. A new derivative
                    map for a batch. 
                    
                */
                map<string, dmat> BATCH_DERIVATIVE;
                
                /**
                     *@brief   
                    *  It will loop on chunks thats created by current and end pointers
                    *  Input:      ### for batch of 3
                    *  Loop over:  # # #
                    * 
                    * @param i   will be point current chunk from input 
                    * @param end will be the end point of batch
                */
                for(int i=current;i<end;i++){
                    /* start of a batch */

                    // Calculating the y_pred
                    map<string,dmat> cache;
                    dmat PREDICTED_OUPUT;
                    tie(PREDICTED_OUPUT,cache) = Feedforward(input[i]);

                    double LOSS = LossesClass::CalculateLoss(output[i],PREDICTED_OUPUT,loss);
                    BATCH_LOSS+=LOSS;

                    dmat derivative_of_loss = LossesClass::CalculateDerivativeLoss(output[i],PREDICTED_OUPUT,loss);
                    BATCH_DERIVATIVE = Backpropagation(
                        derivative_of_loss,
                        cache,
                        BATCH_DERIVATIVE);

                    j++;

                    /* End of one batch */
                }

                // Updating the parameters
                parameters = optimizer.Optimize(
                    BATCH_DERIVATIVE,
                    parameters,
                    j,
                    RecordIndex-1 // Total number of layers
                );


                double AVG_BATCH_LOSS = BATCH_LOSS/(double)j;
                EPOCH_LOSS += AVG_BATCH_LOSS;

                // Printing the progress
                print_progress_bar(batch_number,total_batches,AVG_BATCH_LOSS);

                /*formation of next batch chunk accordingly to batch size.*/
                current = end;
                end = end+batch_size;
                if(end>=input_size){
                    end=input_size;
                }
                batch_number++;
            }/*End of all batches*/

            double AVG_EPOCH_LOSS = EPOCH_LOSS/(double)total_batches;
            losses[epoch]= AVG_EPOCH_LOSS;

            // Ending the line for printing progress in the new line 
            cout<<endl;

        }/*End of one epoch */


        RETURN_MAP["losses"]=losses;
        return RETURN_MAP;
    }

    dmat NeuralNetwork::Predict(dmat input){
        map<string,dmat> cache;
        dmat PREDICTED_OUPUT;
        tie(PREDICTED_OUPUT,cache) = Feedforward(input);
        return PREDICTED_OUPUT;
    }

    arma::field<dmat> NeuralNetwork::Predict(arma::field<dmat> input){
        int input_size=input.size();
        arma::field<dmat> output;
        output.set_size(input_size);
        // loop through each input 
        for(int i=0;i<input_size;i++){
            map<string,dmat> cache;
            dmat PREDICTED_OUPUT;
            tie(PREDICTED_OUPUT,cache) = Feedforward(input[i]);
            output[i]=PREDICTED_OUPUT;
        }
        return output;
    }

    void NeuralNetwork::SaveParameters(string filename){
        // Initiating the file streamer
        ofstream tfStream(filename, std::ios::binary);
        // Iterating over the parameter map
        for (auto &item : parameters) {
            // Writing the name of the dmat matrix
            // for ex W1 or B1 or W2
            tfStream<<item.first<<endl;
            // Inserting unique identifier
            tfStream<<"*/st/*"<<endl;
            // Saving the dmat matrix 
            item.second.save(tfStream,arma::arma_ascii);
            tfStream<<endl;
            // Inserting the unique ending identifier
            tfStream<<"*/en/*"<<endl;
        }
    }

    void NeuralNetwork::LoadParameters(string filename){
        // Initiating the file streamer
        ifstream ifastream(filename,ios::in | ios::out | ios::binary);
        string line;
        string dmatName;

        // iterating over each line from the file
        while (getline(ifastream,line)){
            /* 
               Temperory file to write the value of matrix
               so that we can call the default armadillo's
               load method.
            */
            ofstream tempfStream(".temp", std::ios::binary|std::ofstream::trunc);

            if(line!="*/st/*"){
                dmatName=line;
            }
            // Start of the values of matrix
            if(line=="*/st/*"){
                dmat dmatHold;
                getline(ifastream,line);
                // writing the matrix saved values to the temp file
                while(line!="*/en/*"){
                    tempfStream<<line<<endl;
                    getline(ifastream,line);
                }
                // Creating the input stream to read the temp file
                ifstream tempR(".temp");
                // Loading the values to a dmat matrix
                dmatHold.load(tempR);
                // updating the parameter map
                parameters[dmatName]=dmatHold;
            }
        }
        // Deleting the temp file
        bool status = remove(".temp"); 
    }








};


