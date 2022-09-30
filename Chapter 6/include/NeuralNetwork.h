// NeuralNetwork.h

#include <armadillo>
#include <map>
using namespace std;
using std::map;
using namespace arma;



namespace ArtificialNeuralNetwork{

    class ActivationClass{
        public:
            enum ActivationFunction{
                Linear,
                BinaryStep,
                Relu,
                LeakyReLU,
                Sigmoid,
                Tanh,
            };
        
            static dmat CalculateActivationFunction(dmat matrix,ActivationFunction activation);    
            static dmat LinearFunction(dmat matrix);
            static dmat BinaryStepFunction(dmat matrix);
            static dmat ReluFunction(dmat matrix);
            static dmat LeakyReLUFunction(dmat matrix);
            static dmat SigmoidFunction(dmat matrix);
            static dmat TanhFunction(dmat matrix);

            static dmat CalculateDerivativeActivationFunction(dmat matrix,ActivationFunction activation);    
            static dmat DerivativeLinearFunction(dmat matrix);
            static dmat DerivativeBinaryStepFunction(dmat matrix);
            static dmat DerivativeReluFunction(dmat matrix);
            static dmat DerivativeLeakyReluFunction(dmat matrix);
            static dmat DerivativeSigmoidFunction(dmat matrix);
            static dmat DerivativeTanhFunction(dmat matrix);


    };

    class LossesClass{
        public:
            enum LossFunctions{
                MeanSquareError,
                MeanAbsoluteError,
                BinaryCrossEntropy,
                CategoricalCrossEntropy
            };

            static double CalculateLoss(dmat y_true,dmat y_pred,LossFunctions loss);
            static double MeanSquareErrorFunction(dmat y_true,dmat y_pred);
            static double MeanAbsoluteErrorFunction(dmat y_true,dmat y_pred);
            static double BinaryCrossEntropyFunction(dmat y_true,dmat y_pred);
            static double CategoricalCrossEntropyFunction(dmat y_true,dmat y_pred);

            static dmat CalculateDerivativeLoss(dmat y_true,dmat y_pred,LossFunctions loss);
            static dmat DerivativeMeanSquareErrorFunction(dmat y_true,dmat y_pred);
            static dmat DerivativeMeanAbsoluteErrorFunction(dmat y_true,dmat y_pred);
            static dmat DerivativeBinaryCrossEntropyFunction(dmat y_true,dmat y_pred);
            static dmat DerivativeCategoricalCrossEntropyFunction(dmat y_true,dmat y_pred);

            
    };
  
    class OptimizerClass{
        private:
            double learning_rate;
            double epsilon;
            double rho;
            map<string,dmat> previous_state;
            enum OptimizersEnum{
                GradientDescentEnum,
                AdaGradEnum,
                RMSPropEnum,
            };
            OptimizersEnum SelectedOptimizer;
            dmat CalculateSelectedOptimizer(dmat derivatives, string ref_name);
            dmat CalculateGradientDescent(dmat derivatives);
            dmat CalculateAdaGrad(dmat derivatives);
            dmat CalculateRMSProp(dmat derivatives,string ref_name);

        public:
            OptimizerClass();
            void GradientDescent(double LearningRate);
            void AdaGrad(double LearningRate,double Epsilon);
            void RMSProp(double LearningRate,double Rho,double Epsilon);
            map<string,dmat> Optimize(map<string,dmat> derivatives,map<string,dmat> parameters,int batch_size,int total_layers);
    };
   

    class NeuralNetwork{
        public:
            typedef ActivationClass::ActivationFunction Activations;
            typedef LossesClass::LossFunctions Losses;
        private:
            enum LayerTypes {
                INPUT,
                DENSE,
            };

            /**
             * *************
             * @brief LayerBP 
             * *************
             * struct that provides a way to define each layer and its
             * activations
            */
            struct LayerBP{
                // What is the level of layer in the architecture
                int LayerNumber; 
                // Number of neurons 
                int Neurons; 
                // Input or Hidden (Dense) Layer
                LayerTypes LayerType; 
                // what is the Activation function of layer
                Activations Activation;
                // total number of trainable weights and biases of this layer
                unsigned long int TotalTrainableParameters; 
            };

            /* Initializing InputLayerFlag to keep a check if the input layer
                is defined or not */
            bool InputLayerFlag;

            /* Initializing Records */
            LayerBP* Records;

            /* Declaring Record index to maintain the number of records stored in Records and the number of layers in the neural network, 
            initialized to zero in constructor */
            int RecordIndex; 

            /*This map will contain all the weights and biases values that 
            will be added to network*/
            map<string,dmat> parameters;
            
            void add_to_record(LayerBP layer);
            void initialize_paramaeters(int n_x, int n_y, int pos);
            
            void print_progress_bar(int batch_number, int total_batches,double avg_batch_loss, int barWidth);

            tuple<dmat, map<string,dmat>> Feedforward(dmat input);
            map<string,dmat> Backpropagation(dmat DerivativeLoss, map<string,dmat> cache, map<string,dmat> derivatives);
            

        public:
            NeuralNetwork();
            typedef LayerBP Layer;

            void Add(Layer layer);
            Layer InputLayer(int Neurons);
            Layer DenseLayer(int Neurons, Activations ActivationFunction);
            void PrintMap();
            void Summary();

            map<string,dmat> Train(
                arma::field<dmat> input, 
                arma::field<dmat> output,
                OptimizerClass optimizer,
                int epochs, 
                int batch_size,
                Losses loss
            );

            dmat Predict(dmat input);
            arma::field<dmat> Predict(arma::field<dmat> input);

            void SaveParameters(string filename);
            void LoadParameters(string filename);
    };

};
