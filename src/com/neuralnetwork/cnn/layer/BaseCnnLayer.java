package com.neuralnetwork.cnn.layer;

import com.neuralnetwork.cnn.MNeuron;
import com.neuralnetwork.cnn.filter.IFilter;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

public abstract class BaseCnnLayer implements INeuralNetwork.IMatrixNeuralNetwork
{
    /**
     * the kernel is made of weights from this neuron.
     * the shared neuron contains most of the info necessary to build the layer
     */
    protected MNeuron sharedNeuron;
    protected DenseMatrix64F kernel;

    /**
     * one dimension of the input. assumes input is square
     */
    protected int inputDim;

    /**
     * matrix used to store latest output
     */
    protected DenseMatrix64F output;

    /**
     * the filter used to convolve
     */
    protected IFilter filter;

    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F input)
    {
        output = generateInducedLocalField(input);

        //apply activation function to output
        IActivationFunction phi = sharedNeuron.phi();
        double[] _output = output.getData();

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
        filter.convolve(input, output);

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
