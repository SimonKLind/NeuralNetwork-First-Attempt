/** This is just a file for handling the layers
  * and loading the images */

#include <iostream>
#include <fstream>
#include <string>

#include "Layers.h"

using namespace std;

/** This stuff could probably be done nicer, but whatever... */
string label[10];
unsigned char labels1[10000];
unsigned char *set1[10000];
unsigned char labels2[10000];
unsigned char *set2[10000];
unsigned char labels3[10000];
unsigned char *set3[10000];
unsigned char labels4[10000];
unsigned char *set4[10000];
unsigned char labels5[10000];
unsigned char *set5[10000];
unsigned char testLabels[10000];
unsigned char *test[10000];

void getLabels(){
	ifstream fin;
	fin.open("cifar-10/batches.meta.txt"); // This is just where it's located on my machine...
	if(!fin){
		cout << "ERROR: Label input" << endl;
		return;
	}
	for(int i=0; i<10 && !fin.eof(); i++) getline(fin, label[i]);
	fin.close();
}

/** Gets data from file to passed label array */
void getData(unsigned char *labels, unsigned char **inputs, string filename){
	for(int i=0; i<10000; i++) inputs[i] = new unsigned char[3072]; 
	ifstream fin;
	fin.open("cifar-10/" + filename);
	if(!fin){
		cout << "ERROR: Data input" << endl;
		return;
	}
	for(int i=0; !fin.eof(); i++){
		if(i%3073 == 0) labels[(int)(i/3073)] = fin.get();
		else inputs[(int)(i/3073)][i%3073-1] = fin.get();
	}
	fin.close();
}

void deleteData(){
	for(int i=0; i<10000; ++i){
		delete[] set1[i];
		delete[] set2[i];
		delete[] set3[i];
		delete[] set4[i];
		delete[] set5[i];
		delete[] test[i];
	}
}

/** Cycles through one set of images */
void cycle(unsigned char *labels, unsigned char **inputs, Layer<unsigned char> &first, Layer<double> **layers, int numLayers){
	double *res;
	double loss = 0;
	//double prev = 0;
	int count = 0;
	for(int i=0; i<10000; ++i){ // For each image
		// Forward pass
		res = first.forward(inputs[i], false); // Handle first layer separately
		for(int j=0; j<numLayers-1; ++j) res = layers[j]->forward(res);

		double *temp = res;
		double guess = temp[0];
		int guessIndex = 0;
		res = new double[layers[numLayers-1]->getInWidth()+1]; // Make room for correct label
		for(int j=0; j<layers[numLayers-1]->getInWidth(); ++j){
			if(temp[j] > guess){
				guess = temp[j];
				guessIndex = j;
			}
			res[j] = temp[j];
		}
		if(guessIndex == labels[i]) ++count; 
		delete[] temp;
		res[layers[numLayers-1]->getInWidth()] = labels[i]; // Pass correct label as input[size]
		res = layers[numLayers-1]->forward(res);
		//if((i+1)%100 == 0) cout << "Correct: " << (int)labels[i] << " Guess: " << guessIndex << " Loss: " << *res << endl;
		loss += *res;
		delete res; // Dont really care about loss
		// Forward pass done
		//cout << "Forward done!\n";
		// Backward Pass
		for(int j=numLayers-1; j>=0; --j){
			//cout << j << ": ";
			res = layers[j]->backward(res);
		}
		res = first.backward(res);
		delete[] res; // Dont care about passing any further
		if((i+1)%1000 == 0){
			first.update();
			for(int i=0; i<numLayers; ++i) layers[i]->update();
		}
	}
	cout << "Done Loss: " << loss << " -> " << count/100 << "%" << endl;
	first.update();
	for(int i=0; i<numLayers; ++i) layers[i]->update();
}

/** Runs layers on test set
  * Something in here seems to be wrong, 
  * it always registers exactly 10% even when all train sets 
  * registers higher of lower */
void testNet(Layer<unsigned char> &first, Layer<double> **layers, int numLayers){
	double max;
	double *res;
	int maxIndex;
	int correctCount=0;
	for(int i=0; i<10000; ++i){ // For each image
		// Forward pass
		res = first.forward(test[i], false); // Handle first layer separately
		for(int j=0; j<numLayers-1; ++j) res = layers[j]->forward(res);

		max = res[0];
		maxIndex = 0;
		//cout << max << ", " << res[0] << endl;
		for(int j=1; j<layers[numLayers-2]->getOutWidth(); ++j){
			if(res[j] > max){
				max = res[j];
				maxIndex = j;
			}
			//cout << max << ", " << res[j] << endl;
		}
		//cout << endl;
		if(maxIndex == testLabels[i]){
			++correctCount;
			//cout << i << ": " << maxIndex << " - " << (int)testLabels[i] << " -> " << correctCount << endl << endl;
		}
		delete[] res;
	}
	cout << correctCount/100 << "% correct" << endl;
}

int main(){
	getLabels();
	getData(labels1, set1, "data_batch_1.bin");
	getData(labels2, set2, "data_batch_2.bin");
	getData(labels3, set3, "data_batch_3.bin");
	getData(labels4, set4, "data_batch_4.bin");
	getData(labels5, set5, "data_batch_5.bin");
	getData(testLabels, test, "test_batch.bin");
	
	/* Current layer setup:
	 * ReLU -> BatchNorm -> ReLu -> BatchNorm -> Output -> BatchNorm -> softmax */
	ReLU<unsigned char> u(3072, 100);
	Layer<double> *layers[] = {
		new BatchNorm(100),
		new ReLU<double>(100, 50),
		new BatchNorm(50),
		new Output(50, 10),
		new BatchNorm(10),
		new SoftMax(10)
		//new SVM(10)
	};
	int numLayers = 6;
	
	/* cycles layers through all train sets */
	for(int i=0; i<2; ++i){
		cycle(labels1, set1, u, layers, numLayers);
		cycle(labels2, set2, u, layers, numLayers);
		cycle(labels3, set3, u, layers, numLayers);
		cycle(labels4, set4, u, layers, numLayers);
		cycle(labels5, set5, u, layers, numLayers);
	}
	//cycle(labels1, set1, u, layers, numLayers);
	testNet(u, layers, numLayers);
	deleteData();
	return 0;
}
