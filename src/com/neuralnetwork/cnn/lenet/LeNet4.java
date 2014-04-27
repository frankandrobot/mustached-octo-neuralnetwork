package com.neuralnetwork.convolutional.lenet;

import java.util.Random;

import com.neuralnetwork.convolutional.ConvolutionMap;
import com.neuralnetwork.convolutional.ConvolutionalNetwork;
import com.neuralnetwork.convolutional.FeatureMap;
import com.neuralnetwork.convolutional.MNeuron;
import com.neuralnetwork.convolutional.SubSamplingMap;
import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.SingleLayerNeuralNetwork;
import com.neuralnetwork.core.interfaces.IActivationFunction;

/**
 * LeNet4 (???)
 * Based on a 1990 paper by LeCun: 
 * 
 * Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. 
 * Handwritten digit recognition with a back-propagation network. 
 * Advances in Neural Information Processing Systems 2 (NIPS*89)
 * 
 *
 */
public class LeNet4 
{
    ConvolutionalNetwork CNN;
    
    public LeNet4()
    {
        Random random = new Random();
        
        final IActivationFunction.IDifferentiableFunction phi = new ActivationFunctions.SigmoidUnityFunction(); 
        
        //1st layer
        ConvolutionMap[] H1 = new ConvolutionMap[4];
        for(int j=0; j < H1.length; j++)
        {
            FeatureMap.Builder builder = new FeatureMap.Builder();
            builder.set1DInputSize(28);
            builder.setReceptiveFieldSize(5*5);
            
            double[] weights = new double[5*5+1];
            for(int i=0; i<weights.length; i++)
                weights[i] = random.nextGaussian();
            
            builder.setNeuron(new MNeuron(phi, weights));
            H1[j] = new ConvolutionMap(builder);
        }
        
        //2nd layer
        SubSamplingMap[] H2 = new SubSamplingMap[4];
        for(int j=0; j < H2.length; j++)
        {
            FeatureMap.Builder builder = new FeatureMap.Builder();
            builder.set1DInputSize(24);
            builder.setReceptiveFieldSize(2*2);
            
            double[] weights = new double[2];
            for(int i=0; i<weights.length; i++)
                weights[i] = random.nextGaussian();
            
            builder.setNeuron(new MNeuron(phi, weights));
            H2[j] = new SubSamplingMap(builder);
        }
        
        //3rd layer
        ConvolutionMap[] H3 = new ConvolutionMap[12];
        for(int j=0; j < H3.length; j++)
        {
            FeatureMap.Builder builder = new FeatureMap.Builder();
            builder.set1DInputSize(12);
            builder.setReceptiveFieldSize(5*5);
            
            double[] weights = new double[5*5+1];
            for(int i=0; i<weights.length; i++)
                weights[i] = random.nextGaussian();
            
            builder.setNeuron(new MNeuron(phi, weights));
            H3[j] = new ConvolutionMap(builder);
        }
        
        SubSamplingMap[] H4 = new SubSamplingMap[12];
        for(int j=0; j < H4.length; j++)
        {
            FeatureMap.Builder builder = new FeatureMap.Builder();
            builder.set1DInputSize(8);
            builder.setReceptiveFieldSize(2*2);
            
            double[] weights = new double[2];
            for(int i=0; i<weights.length; i++)
                weights[i] = random.nextGaussian();
            
            builder.setNeuron(new MNeuron(phi, weights));
            H4[j] = new SubSamplingMap(builder);
        }

        //output layer
        SingleLayerNeuralNetwork output = new SingleLayerNeuralNetwork();
        MNeuron[] neurons = new MNeuron[10];
        for(int i=0; i<neurons.length; i++)
        {
        	double[] weights = new double[12*4*4 + 1];
        	for(int j=0; j<weights.length; j++)
        		weights[j] = random.nextGaussian();
        	neurons[i] = new MNeuron(phi, weights);
        }
        output.setNeurons(neurons);
    }
}
