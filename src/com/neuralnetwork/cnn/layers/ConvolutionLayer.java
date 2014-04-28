package com.neuralnetwork.cnn.layers;

import com.neuralnetwork.cnn.MNeuron;
import com.neuralnetwork.cnn.filter.IConvolutionFilter;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

public class ConvolutionLayer implements INeuralNetwork.IMatrixNeuralNetwork
{
    protected MNeuron sharedNeuron;
    protected DenseMatrix64F kernel;
    protected int inputDim;
    protected DenseMatrix64F output;
    protected IConvolutionFilter convolutionFilter;

    public ConvolutionLayer(FeatureMapBuilder builder) throws IllegalArgumentException
    {
        //the neuron aka kernel contains most of the info necessary to build the layer
        sharedNeuron = builder.getSharedNeuron();
        inputDim = builder.getInputDim();
        convolutionFilter = builder.getConvolutionFilter();

        int kernelDim = (int) Math.sqrt(sharedNeuron.getNumberOfWeights()-1);

        if (kernelDim*kernelDim != sharedNeuron.getNumberOfWeights()-1)
            throw new IllegalArgumentException("Kernel must be square");
        if (inputDim < kernelDim)
            throw new IllegalArgumentException("Input can't be smaller than kernel");

        double[] weights = sharedNeuron.getWeights().getData();
        //weights = Arrays.copyOf(weights, weights.length-1); //drop bias
        kernel = new DenseMatrix64F(kernelDim, kernelDim, true, weights);

        convolutionFilter.setKernel(kernel);

        output = new DenseMatrix64F(
                inputDim-kernelDim+1,
                inputDim-kernelDim+1);
    }

    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F input)
    {
        output = generateInducedLocalField(input);

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
    public DenseMatrix64F generateInducedLocalField(DenseMatrix64F input)
    {
        //first the weights
        convolutionFilter.convolve(input, output);
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

    public int getInputDim() { return inputDim; }
}