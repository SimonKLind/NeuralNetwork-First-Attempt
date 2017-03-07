#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#include <cmath>
#include <iostream>

std::random_device rd;
std::mt19937 rando(rd());

const double LEARN_RATE = 10;
const double REGULARIZATION_STRENGTH = 0.01;

template <typename inType>
class Layer{

protected:
	int inW, outW;
	double *latestIn; // Latest inputs, will add up gradients

public:

	Layer(int inWidth, int outWidth): inW(inWidth), outW(outWidth){
		latestIn = new double[inWidth];
	}

	virtual double* forward(inType *input, bool del = true) = 0;
	virtual double* backward(double *grad) = 0;
	virtual void update(){}

	int getInWidth(){
		return inW;
	}

	int getOutWidth(){
		return outW;
	}

	virtual ~Layer(){
		delete[] latestIn;
	}
};

/* -------------------------------------------------------- */

class BatchNorm: public Layer<double>{

	double mean;
	double variance;

	double getVariance(){
		variance = 0;
		for(int i=0; i<this->inW; i++) variance += (this->latestIn[i]-mean)*(this->latestIn[i]-mean);
		variance /= this->inW;
	}
	
public:

	BatchNorm(int inWidth): Layer(inWidth, inWidth) {}

	double* forward(double *input, bool del = true){
		//std::cout << "Going through BatchNorm: ";
		mean = 0;
		for(int i=0; i<this->inW; ++i){
			//std::cout << input[i] << ", ";
			this->latestIn[i] = input[i];
			mean+=input[i];
		}
		mean /= this->inW;
		getVariance();

		double *output = new double[this->outW];
		//std::cout << "\n\nExiting BatchNorm: ";
		for(int i=0; i<inW; ++i){
			output[i] = (input[i]-mean)/sqrt(variance);
			//std::cout << output[i] << ", ";
		}
		//std::cout << "\n\n";

		if(del) delete[] input;
		return output;
	}

	double* backward(double *grad){
		double iw = this->inW;
		double *out = new double[this->inW];
		for(int i=0; i<this->inW; ++i){
			out[i] = grad[i]*((1.0-1.0/iw)/sqrt(variance)-((2.0/iw-4.0/(iw*iw))*(this->latestIn[i]-mean)*(this->latestIn[i]-mean))/(2.0*pow(variance, 1.5)));
			//std::cout << out[i] << "; " << grad[i] << ", ";
		} // This gradient is just a fucking monster
		//std::cout << "\n\n";
		delete[] grad;
		return out;
	}
};

/* -------------------------------------------------------- */

class SoftMax: public Layer<double>{

	double *exps;
	double total;

	double sum(int skip = -1){ // If an index is specified, it will be skipped
		double val = 0;
		for(int i=0; i<this->inW; i++) if(i != skip) val+=exps[i];
		return val;
	}

public:

	int correct = -1;

	SoftMax(int inWidth): Layer(inWidth, 1){
		exps = new double[inWidth];
	}

	double* forward(double *input, bool del = true){
		//std::cout << "Going through SoftMax: ";
		correct = input[this->inW];
		if(correct == -1) return nullptr;
		for(int i=0; i<this->inW; i++){
			this->latestIn[i]=input[i];
			exps[i] = exp(input[i]);
			//std::cout << input[i] << ", ";
		}
		//std::cout << "\n\nExiting SoftMax: ";
		total = sum();
		double *output = new double;
		*output = log(total)-input[correct];
		if(del) delete[] input;
		//std::cout << *output << "; " << log(total)-input[correct] << "\n\n";
		return output;
	}

	double* backward(double *grad){
		grad = new double[this->inW];
		for(int i=0; i<this->inW; i++){
			grad[i] = exps[i]/total;
			//std::cout << grad[i] << "; " << exps[i] << ", ";
		}
		//std::cout << "\n\n";
		grad[correct]-=1;
		return grad;
	}

	virtual ~SoftMax(){
		delete[] exps;
	}

};

/* -------------------------------------------------------- */

class SVM: public Layer<double>{

	char *gradMult;

	int correct = -1;

	double clamp(double val){
		return (val >= 0) ? val : -val;
	}

public:
	SVM(int inWidth): Layer(inWidth, 1){
		gradMult = new char[inWidth];
	}

	double* forward(double *input, bool del = true){
		correct = input[this->inW];
		if(correct == -1) return nullptr;
		double *out = new double;
		double current;
		for(int i=0; i<this->inW; ++i){
			this->latestIn[i] = input[i];
			if(i != correct){
				current = clamp(input[i]-input[correct]+1);
				(current>0) ? gradMult[i] = 1 : gradMult[i] = 0;
				*out+=current;
			}
		}
		if(del) delete[] input;
		return out;
	}

	double* backward(double *grad){
		grad = new double[this->inW];
		for(int i=0; i<this->inW; ++i){
			(i != correct && gradMult[i] != 0) ? grad[i] = this->latestIn[i] : grad[i] = 0;
		}
		return grad;
	}
};

/* -------------------------------------------------------- */

template <typename inType>
class WeightLayer: public Layer<inType>{

protected:
	double **weights; // All weights
	double **gradients; // All gradients

public:

	WeightLayer(int inWidth, int outWidth): Layer<inType>(inWidth, outWidth){
		weights = new double*[outWidth];
		gradients = new double*[outWidth];
		for(int i=0; i<outWidth; i++){
			weights[i] = new double[inWidth+1];
			gradients[i] = new double[inWidth+1];
			for(int j=0; j<=inWidth; j++){
				weights[i][j] = (int)rando()%inWidth*sqrt((double)2/inWidth);
				gradients[i][j] = 0;
			}
		}
	}

	virtual double* forward(inType *input, bool del = true) = 0;
	virtual double* backward(double *grad) = 0;

	void update(){
		for(int i=0; i<this->outW; i++){
			for(int j=0; j<this->inW; j++){
				//std::cout << weights[i][j] << " -> ";
				weights[i][j] -= gradients[i][j]*LEARN_RATE+weights[i][j]*REGULARIZATION_STRENGTH;
				//std::cout << weights[i][j] << ", " << gradients[i][j] << std::endl;
				gradients[i][j] = 0;
			}
			weights[i][this->inW] -= gradients[i][this->inW];
		}
	}

	virtual ~WeightLayer(){
		for(int i=0; i<this->outW; i++){
			delete[] weights[i];
			delete[] gradients[i];
		}
		delete[] gradients;
		delete[] weights;
	}
};

/* -------------------------------------------------------- */

template <typename inType>
class ReLU: public WeightLayer<inType>{

	char *gradMult; // Checks if relu activates, kills gradient if not

	double clamp(double val){
		if(val > 0) return val;
		return 0;
	}

public:

	ReLU(int inWidth, int outWidth): WeightLayer<inType>(inWidth, outWidth){
		gradMult = new char[outWidth];
		for(int i=0; i<outWidth; i++) gradMult[i] = 0;
	}

	double* forward(inType *input, bool del = true){
		//std::cout << "Going through ReLU: ";
		double *output = new double[this->outW];
		for(int i=0; i<this->inW; i++){
			//std::cout << input[i] << ", ";
			this->latestIn[i] = input[i];
		}
		//std::cout << "\n\nExiting ReLU: ";
		for(int i=0; i<this->outW; i++){
			for(int j=0; j<this->inW; j++) output[i] += input[j]*this->weights[i][j];
			output[i] += this->weights[i][this->inW];
			output[i] = clamp(output[i]);
			(output[i] > 0) ? gradMult[i] = 1 : gradMult[i] = 0;
			//std::cout << output[i] << ", ";
		}
		//std::cout << "\n\n";
		if(del) delete[] input;
		return output;
	}

	double* backward(double *grad){
		//double currentGradients[this->outW][this->inW];
		for(int i=0; i<this->outW; i++){ // Adding up gradients
			//std::cout << grad[i] << ", ";
			if(gradMult[i] != 0) for(int j=0; j<this->inW; j++) this->gradients[i][j] += this->latestIn[j]*grad[i];
			this->gradients[i][this->inW] += grad[i];
		}
		//std::cout << "\n\n";

		//std::cout << "Hello " << this->inW << std::endl;
		double *out = new double[this->inW]; // Summarize gradients to pass to previous level
		//std::cout << "Hello 2\n";
		for(int i=0; i<this->inW; i++) out[i] = 0;
		for(int i=0; i<this->outW; i++){
			if(gradMult[i] != 0) for(int j=0; j<this->inW; j++) out[j]+=this->weights[i][j]*grad[i];
		}
		
		delete[] grad;	
		return out;
	}

	virtual ~ReLU(){
		delete[] gradMult;
	}
};

/* -------------------------------------------------------- */

class Output: public WeightLayer<double>{

public:

	Output(int inWidth, int outWidth): WeightLayer(inWidth, outWidth){}

	double* forward(double *input, bool del = true){
		//std::cout << "Going through Output: ";
		double *output = new double[this->outW];
		for(int i=0; i<this->inW; i++){
			//std::cout << input[i] << ", ";
			this->latestIn[i] = input[i];
		}
		//std::cout << "\n\nExiting Output: ";
		for(int i=0; i<this->outW; i++){
			for(int j=0; j<this->inW; j++) output[i] += input[j]*this->weights[i][j];
			output[i] += this->weights[i][this->inW];
			//std::cout << output[i] << ", ";
		}
		//std::cout << "\n\n";
		if(del) delete[] input;
		return output;
	}

	double* backward(double *grad){
		for(int i=0; i<this->outW; i++){
			//std::cout << grad[i] << ", ";
			for(int j=0; j<this->inW; j++) this->gradients[i][j] += grad[i]*this->latestIn[j];
			this->gradients[i][this->inW] += grad[i];
		}
		//std::cout << "\n\n";

		double *out = new double[this->inW];
		for(int i=0; i<this->inW; i++) out[i] = 0;
		for(int i=0; i<this->outW; i++){
			for(int j=0; j<this->inW; j++) out[j] += this->weights[i][j]*grad[i];
		}

		delete[] grad;
		return out;
	}

};

/* -------------------------------------------------------- */

#endif