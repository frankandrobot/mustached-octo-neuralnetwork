package com.neuralnetwork.convolutional.convolutionmap;

import com.neuralnetwork.convolutional.MNeuron;
import com.neuralnetwork.convolutional.filter.IConvolutionFilter;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Arrays;
import java.util.Iterator;

public class SubSamplingLayer implements INeuralNetwork.IMatrixNeuralNetwork
{
    protected MNeuron sharedNeuron;
    protected DenseMatrix64F kernel;
    protected int inputDim;
    protected DenseMatrix64F output;
    protected IConvolutionFilter samplingFilter;

    public SubSamplingLayer(FeatureMapBuilder builder) throws IllegalArgumentException
    {
        //the neuron aka kernel contains most of the info necessary to build the layer
        sharedNeuron = builder.sharedNeuron;
        inputDim = builder.inputDim;
        samplingFilter = builder.convolutionFilter;

        int kernelDim = (int) Math.sqrt(sharedNeuron.getNumberOfWeights()-1);

        if (kernelDim*kernelDim != sharedNeuron.getNumberOfWeights()-1)
            throw new IllegalArgumentException("Kernel must be square");
        if (inputDim < kernelDim)
            throw new IllegalArgumentException("Input can't be smaller than kernel");
        if (inputDim % kernelDim != 0)
            throw new IllegalArgumentException("Input size must be a multiple of the kernel size");

        double[] weights = sharedNeuron.getWeights().getData();
        double weight = weights[0];
        for(int i=1; i<weights.length; i++)
            if (weights[i] != weight)
                throw new IllegalArgumentException("Kernel weights must all be equal");

        //weights = Arrays.copyOf(weights, weights.length-1); //drop bias
        kernel = new DenseMatrix64F(kernelDim, kernelDim, true, weights);
        samplingFilter.setKernel(kernel);

        output = new DenseMatrix64F(
                inputDim/kernelDim,
                inputDim/kernelDim);
    }

    @Override
    public DenseMatrix64F constructOutput(DenseMatrix64F input)
    {
        output = constructInducedLocalField(input);

        //apply activation function to output
        double[] _output = output.getData();
        IActivationFunction phi = sharedNeuron.phi();
        for(int i=0; i<_output.length; i++)
        {
            _output[i] = phi.apply(_output[i]);
        }
        return output;
    }

    @Override
    public DenseMatrix64F constructInducedLocalField(DenseMatrix64F input)
    {
        //first the weights
        samplingFilter.convolve(input, output);
        //now the biases
        double bias = sharedNeuron.getBias();
        double[] o = output.getData();
        for(int i=0; i<o.length; i++)
        {
            o[i] += bias;
        }
        return output;
    }

    @Override
    public DenseMatrix64F getOutput()
    {
        return output;
    }

    @Override
    public int getNumberOfNeurons()
    {
        return 1;
    }

    @Override
    public MNeuron getNeuron(int neuron)
    {
        return sharedNeuron;
    }

    @Override
    public Iterator<MNeuron> iterator()
    {
        throw new NotImplementedException();
    }
}
